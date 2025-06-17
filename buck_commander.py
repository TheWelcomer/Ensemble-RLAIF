from huggingface_hub import login
import os
login(token=os.getenv('HF_API_KEY'))
os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
os.environ["WANDB_PROJECT"] = "AI-Scribing"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
system_prompt = """
# AI Clinical Note Generator

**Role**: You are an expert clinical AI assistant specializing in generating accurate and useful medical documentation.

**Task**: Based on the provided medical transcript, generate a clinical note that strictly adheres to the instructions below.

---

### CRITICAL INSTRUCTIONS

1.  **Strict Transcript Adherence**: The note must be generated **using ONLY information explicitly stated** within the provided transcript.
    -   **DO NOT** infer information, add details not mentioned (even if clinically likely), or make assumptions. Adherence to the source transcript is the highest priority.
    -   **DO NOT** include any factual errors or hallucinated details.

2.  **Clinical Relevance**: Include all clinically important information from the transcript, but be concise. Omit conversational filler, non-medical chit-chat, and redundant phrases.

3.  **Formatting and Structure**:
    -   The output must be clear, well-organized, and easy for a healthcare provider to read.
    -   Use standard, unambiguous medical abbreviations where appropriate for conciseness.
    -   **DO NOT** include any extraneous text. The output must contain **ONLY** the clinical note itself, without any introductory sentences, concluding remarks, or disclaimers.

4.  **Style Specification**:
    {}
"""
from datasets import load_dataset
dataset = load_dataset("The-Welcomer/scribing-train-dataset-batched", split = "train")
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt.format(x["generator_style_sentence"])},
        {"role": "user",   "content": x["dialogue"]},
    ],
    "ground_truths" : x["google/gemini-2.5-pro-preview"],
    "structure_sentences" : x["evaluator_style_sentence"],
})
from unsloth import FastLanguageModel
import torch
max_seq_length = 7000
lora_rank = 128

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional
import torch
import traceback
import openai
import csv

GEMINI_EVAL_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_NEMOTRON_EVAL_MODEL_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1"
OPENROUTER_DEEPSEEK_EVAL_MODEL_NAME = "deepseek/deepseek-chat-v3-0324"
OPENROUTER_GEMINI_EVAL_MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"

SCORE_MAP = {
    "X_SUBSTANTIALLY": -1.0,
    "X_CLEARLY": -0.6,
    "X_SLIGHTLY": -0.2,
    "NEUTRAL": 0.0,
    "Y_SLIGHTLY": 0.2,
    "Y_CLEARLY": 0.6,
    "Y_SUBSTANTIALLY": 1.0,
}

def versus(structure_sentence, transcript, ft_note, base_note, ft_note_first):
    def evaluate_single(evaluator):
        with ThreadPoolExecutor() as executor:
            ft_first_thread = executor.submit(evaluator.evaluate, structure_sentence, transcript, ft_note, base_note)
            base_first_thread = executor.submit(evaluator.evaluate, structure_sentence, transcript, base_note, ft_note)
            ft_first = ft_first_thread.result()
            base_first = base_first_thread.result()
        if base_first in SCORE_MAP and ft_first in SCORE_MAP:
            ft_first_score = SCORE_MAP[ft_first]
            base_first_score = -SCORE_MAP[base_first]
            final_score = (base_first_score + ft_first_score) / 2
            return final_score
        if not base_first:
            print(f"Warning (versus): Evaluator returned '{base_first}', which is not a recognized X_DEGREE format in SCORE_MAP. Score contribution will be 0.")
        if not ft_first:
            print(f"Warning (versus): Evaluator returned '{ft_first}', which is not a recognized X_DEGREE format in SCORE_MAP. Score contribution will be 0.")
        return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_single, EVALUATORS_LIST))
        none_val = 0
        num_real = 0

        for result in results:
            if result is not None:
                none_val += result
                num_real += 1

        if num_real == 0:
            return None

        for i in range(len(results)):
            if results[i] is None:
                results[i] = none_val / num_real

        return results

class BaseEvaluator:
    def __init__(self, api_key_env_var: str, model_name: str):
        self.model_name = model_name
        self.api_key = None
        try:
            self.api_key = os.getenv(api_key_env_var)
        except ImportError:
            pass
        if not self.api_key:
            self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"API key for {self.__class__.__name__} (env var: {api_key_env_var}) not found."
            )

    def _construct_judge_prompt(self, structure_sentence: str, transcript: str, response_a: str, response_b: str) -> str:
        return f"""
# Clinical Note Evaluator

You are evaluating two clinical notes based on a medical transcript. Your goal is to determine which note would be more useful for patient care in a real clinical setting.

## Evaluation Criteria

**Accuracy**: Does the note contain only information from the transcript? Are there any factual errors or hallucinated details?

**Completeness**: Does the note include all clinically important information from the transcript?

**Clinical Utility**: Is the note clear, well-organized, and easy for healthcare providers to use? Does it follow standard medical documentation practices?

{structure_sentence}

## Instructions

1. **Evaluate Note X** against each criterion
2. **Evaluate Note Y** against each criterion
3. **Compare the notes** and determine which would be more beneficial for patient care
4. **Provide your reasoning** explaining your choice and the magnitude of difference

## Response Format

End your response with exactly one of these choices on a new line based on the overall benefit for patient care:

- `CHOICE: X_SLIGHTLY` - Note X is somewhat better (noticeable but limited advantage)
- `CHOICE: X_CLEARLY` - Note X is clearly better (significant, unambiguous advantage)
- `CHOICE: X_SUBSTANTIALLY` - Note X is much better (major, decisive advantage)
- `CHOICE: NEUTRAL` - Notes are roughly equivalent in clinical value
- `CHOICE: Y_SLIGHTLY` - Note Y is somewhat better (noticeable but limited advantage)
- `CHOICE: Y_CLEARLY` - Note Y is clearly better (significant, unambiguous advantage)
- `CHOICE: Y_SUBSTANTIALLY` - Note Y is much better (major, decisive advantage)

Focus on practical clinical value. Consider which note a healthcare provider would find more useful, accurate, and actionable for patient care.

--- NOW EVALUATE THE FOLLOWING INPUT ---

<input>
    <transcript>
        {transcript}
    </transcript>
    <responses>
        <response_X>
            {response_b}
        </response_X>
        <response_Y>
            {response_a}
        </response_Y>
    </responses>
</input>
"""

    def _parse_response(self, llm_output: str) -> Optional[str]:
        match = re.search(r"(?:\*\*)?CHOICE(?:\*\*)?:\s*(X_(?:SLIGHTLY|CLEARLY|SUBSTANTIALLY)|Y_(?:SLIGHTLY|CLEARLY|SUBSTANTIALLY)|NEUTRAL)", llm_output, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
        log_output_display = llm_output
        if len(log_output_display) > 1000:
            log_output_display = log_output_display
        print(f"Warning ({self.__class__.__name__}): Could not parse choice in 'CHOICE: X_DEGREE' format from LLM output. Raw LLM output snippet: '{log_output_display}'")
        return None

    def evaluate(self, transcript: str, response_a: str, response_b: str) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement the evaluate method.")

class GeminiConcurrentUserEvaluator(BaseEvaluator):
    def __init__(self, api_key_env_var: str = "GEMINI_API_KEY", model_name: str = GEMINI_EVAL_MODEL_NAME):
        super().__init__(api_key_env_var, model_name)
        self.api_url = GEMINI_API_URL_TEMPLATE.format(model_name=self.model_name)
        print(f"GeminiConcurrentUserEvaluator initialized with model: {self.model_name}")

    def evaluate(self, structure_sentence: str, transcript: str, response_a: str, response_b: str) -> Optional[str]:
        judge_prompt_text = self._construct_judge_prompt(structure_sentence, transcript, response_a, response_b)
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": judge_prompt_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 20000
            }
        }
        response_obj = None
        llm_output_text_for_parsing = None

        try:
            full_api_url = f"{self.api_url}?key={self.api_key}"
            response_obj = requests.post(full_api_url, headers=headers, json=payload, timeout=240)
            response_obj.raise_for_status()

            if not response_obj.content:
                print(f"Error (GeminiConcurrentUserEvaluator): Empty response content from API for model {self.model_name}.")
                return None
            try:
                response_data = response_obj.json()
            except requests.exceptions.JSONDecodeError:
                print(f"Error (GeminiConcurrentUserEvaluator): Non-JSON response from API for model {self.model_name}. Status: {response_obj.status_code}. Response text: {response_obj.text[:500]}")
                return None

            if 'promptFeedback' in response_data:
                prompt_feedback = response_data['promptFeedback']
                if 'blockReason' in prompt_feedback:
                    block_reason = prompt_feedback.get('blockReason')
                    safety_ratings_feedback = prompt_feedback.get('safetyRatings', [])
                    print(f"Warning (GeminiConcurrentUserEvaluator): Prompt was blocked for model {self.model_name}. Reason: {block_reason}. Safety Feedback: {safety_ratings_feedback}. Full Feedback: {prompt_feedback}")
                    return None

            if 'candidates' not in response_data or not response_data['candidates']:
                err_msg = f"Error (GeminiConcurrentUserEvaluator): 'candidates' field missing or empty in response from model {self.model_name}."
                if 'promptFeedback' in response_data:
                     err_msg += f" Prompt Feedback: {response_data['promptFeedback']}"
                err_msg += f" Response snippet: {str(response_data)[:500]}"
                print(err_msg)
                return None

            candidate = response_data['candidates'][0]
            finish_reason = candidate.get('finishReason')
            token_count = candidate.get('tokenCount')
            safety_ratings_candidate = candidate.get('safetyRatings', [])

            if 'content' not in candidate or \
               'parts' not in candidate.get('content', {}) or \
               not isinstance(candidate['content'].get('parts'), list) or \
               not candidate['content']['parts']:
                print(f"Error (GeminiConcurrentUserEvaluator): 'content' or 'parts' structure malformed or empty in candidate from model {self.model_name}. "
                      f"Finish Reason: {finish_reason}. Token Count: {token_count}. Candidate Safety Ratings: {safety_ratings_candidate}. "
                      f"Full Candidate: {str(candidate)[:500]}")
                return None

            first_part = candidate['content']['parts'][0]
            if 'text' not in first_part:
                print(f"Error (GeminiConcurrentUserEvaluator): 'text' missing in first part of candidate content from model {self.model_name}. "
                      f"Finish Reason: {finish_reason}. Token Count: {token_count}. Candidate Safety Ratings: {safety_ratings_candidate}. "
                      f"Full Candidate: {str(candidate)[:500]}")
                return None

            llm_output_text_for_parsing = first_part['text']

            if not llm_output_text_for_parsing.strip():
                print(f"Warning (GeminiConcurrentUserEvaluator): Extracted LLM output is empty or whitespace for model {self.model_name}. "
                      f"Finish Reason: {finish_reason}. Token Count: {token_count}. Candidate Safety Ratings: {safety_ratings_candidate}. "
                      f"Full Candidate: {str(candidate)[:500]}")
                return None

            if finish_reason and finish_reason != "STOP":
                print(f"Warning (GeminiConcurrentUserEvaluator): Candidate finishReason is '{finish_reason}' (not 'STOP') for model {self.model_name}. "
                      f"Output might be incomplete. Token Count: {token_count}. Candidate Safety Ratings: {safety_ratings_candidate}. "
                      f"Full Candidate: {str(candidate)[:500]}")

            return self._parse_response(llm_output_text_for_parsing)

        except requests.exceptions.Timeout:
            print(f"Error (GeminiConcurrentUserEvaluator): Request timed out for model {self.model_name}.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error (GeminiConcurrentUserEvaluator): API request error for model {self.model_name}: {e}")
            if response_obj: print(f"Response status: {response_obj.status_code}, Content: {response_obj.text[:500]}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            err_msg = f"Error (GeminiConcurrentUserEvaluator): API response parsing error for model {self.model_name}: {e}."
            if response_obj and hasattr(response_obj, 'text'):
                err_msg += f" Raw Response snippet: {response_obj.text[:500]}"
            else:
                err_msg += " No response object or text available."
            print(err_msg)
            return None
        except Exception as e:
            print(f"Error (GeminiConcurrentUserEvaluator): Unexpected error during evaluation for model {self.model_name}: {e}")
            traceback.print_exc()
            return None

class OpenRouterEvaluator(BaseEvaluator):
    def __init__(self, api_key_env_var: str = "OPENROUTER_API_KEY", model_name: str = None):
        super().__init__(api_key_env_var, model_name)
        self.client = None
        if not openai:
            print(f"Error (OpenRouterEvaluator): OpenAI library not available, which is needed for OpenRouter. Cannot initialize client for model {self.model_name}.")
            return
        try:
            self.client = openai.OpenAI(
                base_url=OPENROUTER_API_URL,
                api_key=self.api_key
            )
            print(f"OpenRouterEvaluator initialized for model: {self.model_name} via OpenRouter API.")
        except Exception as e:
            print(f"Warning (OpenRouterEvaluator): Could not initialize OpenAI client for OpenRouter during __init__. API key might be missing or invalid. Error: {e}")

    def evaluate(self, structure_sentence: str, transcript: str, response_a: str, response_b: str) -> Optional[str]:
        if not self.client:
            if self.api_key and openai:
                try:
                    self.client = openai.OpenAI(
                        base_url=OPENROUTER_API_URL,
                        api_key=self.api_key
                    )
                    print(f"OpenRouterEvaluator: Re-initialized OpenAI client for OpenRouter (model: {self.model_name}).") # Changed class name in log
                except Exception as e:
                    print(f"Error (OpenRouterEvaluator): Failed to re-initialize OpenAI client for OpenRouter in evaluate. Error: {e}") # Changed class name in log
                    return None
            else:
                print(f"Error (OpenRouterEvaluator): OpenAI client for OpenRouter not initialized and no API key or library. Skipping evaluation for model {self.model_name}.") # Changed class name in log
                return None

        judge_prompt_text = self._construct_judge_prompt(structure_sentence, transcript, response_a, response_b)
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
        x_title = os.getenv("OPENROUTER_X_TITLE", "")
        extra_headers = {}
        if http_referer: extra_headers["HTTP-Referer"] = http_referer
        if x_title: extra_headers["X-Title"] = x_title

        response_obj = None
        try:
            response_obj = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": judge_prompt_text}],
                temperature=0.2,
                max_tokens=8192,
                timeout=240,
                logprobs=True,
                top_logprobs=10,
                extra_headers=extra_headers if extra_headers else None
            )

            if not response_obj.choices or not response_obj.choices[0].message or \
               not hasattr(response_obj.choices[0].message, 'content') or not response_obj.choices[0].message.content:
                err_msg = (f"Error (OpenRouterEvaluator): Malformed response structure or empty content from model {self.model_name} via OpenRouter. "
                           f"Finish Reason: {response_obj.choices[0].finish_reason if response_obj and response_obj.choices else 'N/A'}. "
                           f"Full Response (obj): {str(response_obj)[:500]}")
                print(err_msg)
                return None

            llm_output = response_obj.choices[0].message.content
            finish_reason = response_obj.choices[0].finish_reason

            if not llm_output.strip():
                print(f"Warning (OpenRouterEvaluator): Extracted LLM output is empty or whitespace for model {self.model_name} via OpenRouter. "
                      f"Finish Reason: {finish_reason}. Usage: {response_obj.usage}")
                return None

            if finish_reason and finish_reason != "stop":
                print(f"Warning (OpenRouterEvaluator): Candidate finish_reason is '{finish_reason}' (not 'stop') for model {self.model_name} via OpenRouter. "
                      f"Output might be incomplete. Usage: {response_obj.usage}")

            return self._parse_response(llm_output)

        except openai.APIConnectionError as e:
            print(f"Error (OpenRouterEvaluator): API connection error for model {self.model_name} via OpenRouter: {e}")
        except openai.RateLimitError as e:
            print(f"Error (OpenRouterEvaluator): Rate limit exceeded for model {self.model_name} via OpenRouter: {e}")
        except openai.AuthenticationError as e:
            print(f"Error (OpenRouterEvaluator): Authentication error (check OPENROUTER_API_KEY) for model {self.model_name} via OpenRouter: {e}")
            self.client = None
        except openai.BadRequestError as e:
            err_body_str = "N/A"
            if e.response and hasattr(e.response, 'text'):
                 err_body_str = e.response.text[:500]
            print(f"Error (OpenRouterEvaluator): Bad request error for model {self.model_name} via OpenRouter: {e}. Response body: {err_body_str}")
        except openai.APITimeoutError:
            print(f"Error (OpenRouterEvaluator): Request timed out for model {self.model_name} via OpenRouter.")
        except openai.APIStatusError as e:
            print(f"Error (OpenRouterEvaluator): API status error {e.status_code} for model {self.model_name} via OpenRouter: {e.message}")
        except Exception as e:
            print(f"Error (OpenRouterEvaluator): Unexpected error during evaluation for model {self.model_name} via OpenRouter: {e}")
            traceback.print_exc()
        return None

EVALUATORS_LIST: List[BaseEvaluator] = []
def initialize_evaluators():
    global EVALUATORS_LIST
    EVALUATORS_LIST = []
    print("Initializing evaluators...")

    evaluator_configs = [
        {"class": OpenRouterEvaluator, "path": OPENROUTER_NEMOTRON_EVAL_MODEL_NAME, "key_env_var": "OPENROUTER_API_KEY", "name": "Nemotron", "lib": openai},
        {"class": OpenRouterEvaluator, "path": OPENROUTER_DEEPSEEK_EVAL_MODEL_NAME, "key_env_var": "OPENROUTER_API_KEY", "name": "Deepseek V3", "lib": openai},
        {"class": OpenRouterEvaluator, "path": OPENROUTER_GEMINI_EVAL_MODEL_NAME, "key_env_var": "OPENROUTER_API_KEY", "name": "Gemini Flash", "lib": openai},
    ]

    for config in evaluator_configs:
        evaluator_class = config["class"]
        model_path = config.get("path")
        key_env_var = config["key_env_var"]
        evaluator_name = config["name"]
        required_lib = config.get("lib", True)

        if not required_lib:
            print(f"INFO: {evaluator_name} evaluator skipped because its required library is not available.")
            continue

        api_key_present = False
        try:
            if os.getenv(key_env_var):
                api_key_present = True
        except ImportError:
            if os.getenv(key_env_var):
                api_key_present = True

        if not api_key_present:
            print(f"INFO: API key '{key_env_var}' not found. Skipping {evaluator_name} evaluator.")
            continue

        try:
            if model_path:
                judge = evaluator_class(model_name=model_path, api_key_env_var=key_env_var)
            else:
                judge = evaluator_class(api_key_env_var=key_env_var)

            EVALUATORS_LIST.append(judge)
        except ValueError as e:
            print(f"INFO: Could not initialize {evaluator_name} (likely API key '{key_env_var}' missing or error during BaseEvaluator init): {e}")
        except Exception as e:
            print(f"UNEXPECTED ERROR initializing {evaluator_name}: {e}")
            traceback.print_exc()


    if not EVALUATORS_LIST:
        print("Warning: EVALUATORS_LIST is empty after initialization attempt. No evaluators are available.")
    else:
        active_evaluators = [f"{type(e).__name__}({e.model_name})" for e in EVALUATORS_LIST]
        print(f"Successfully prepared {len(EVALUATORS_LIST)} evaluators: {active_evaluators}")

initialize_evaluators()
import asyncio
import wandb
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import statistics
import math
import shutil

per_evaluator_ema_z_score_stats = {}

aggregate_ema_z_score_stats = {
    's_b_intermediate_z': {'mean': 0.0, 'variance': 1.0, 'count': 0},
    's_ft_intermediate_z': {'mean': 0.0, 'variance': 1.0, 'count': 0},
}

EMA_ALPHA_INITIAL = 0.5
EMA_INITIAL_PHASE_STEPS = 5

EMA_ALPHA_PER_EVAL_Z = 0.10
EMA_ALPHA_AGGREGATE_Z = 0.05

EMA_EPSILON = 1e-8

CURRENT_EVALUATOR_KEY_FOR_EMA = None

last_logged_optimizer_step_for_ema = -1

def _update_ema_z_score_and_normalize_core_with_warmup(
    scores, stats_dict_key, ema_storage,
    alpha_initial, alpha_long_term, initial_phase_steps,
    epsilon
):
    if not scores: return []
    current_batch_mean = statistics.mean(scores)
    if len(scores) > 1: current_batch_variance = statistics.variance(scores)
    else: current_batch_variance = 0.0

    stats = ema_storage.get(stats_dict_key)
    current_count = stats['count'] if stats else 0
    updated_count = current_count + 1

    if current_count == 0:
        new_mean = current_batch_mean
        new_variance = current_batch_variance if current_batch_variance > epsilon else 1.0
    else:
        alpha = alpha_initial if updated_count <= initial_phase_steps else alpha_long_term
        new_mean = alpha * current_batch_mean + (1 - alpha) * stats['mean']
        safe_batch_variance = current_batch_variance if current_batch_variance >= 0 else 0.0
        new_variance = alpha * safe_batch_variance + (1 - alpha) * stats['variance']
        new_variance = max(new_variance, epsilon)

    ema_storage[stats_dict_key] = {'mean': new_mean, 'variance': new_variance, 'count': updated_count}

    std_dev = math.sqrt(new_variance)
    if std_dev < epsilon: std_dev = math.sqrt(epsilon)
    normalized_scores = [(s - new_mean) / std_dev for s in scores]
    return normalized_scores

def normalize_stage1_ema_z_score_per_evaluator(scores_list):
    global CURRENT_EVALUATOR_KEY_FOR_EMA, per_evaluator_ema_z_score_stats
    global EMA_ALPHA_INITIAL, EMA_ALPHA_PER_EVAL_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON

    if CURRENT_EVALUATOR_KEY_FOR_EMA is None:
        print("Warning: CURRENT_EVALUATOR_KEY_FOR_EMA not set for Stage 1. Returning raw scores.")
        return scores_list

    return _update_ema_z_score_and_normalize_core_with_warmup(
        scores_list, CURRENT_EVALUATOR_KEY_FOR_EMA, per_evaluator_ema_z_score_stats,
        EMA_ALPHA_INITIAL, EMA_ALPHA_PER_EVAL_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON
    )

def normalize_stage2_ema_z_score_aggregate_base(scores_list):
    global aggregate_ema_z_score_stats
    global EMA_ALPHA_INITIAL, EMA_ALPHA_AGGREGATE_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON
    return _update_ema_z_score_and_normalize_core_with_warmup(
        scores_list, 's_b_intermediate_z', aggregate_ema_z_score_stats,
        EMA_ALPHA_INITIAL, EMA_ALPHA_AGGREGATE_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON
    )

def normalize_stage2_ema_z_score_aggregate_ft(scores_list):
    global aggregate_ema_z_score_stats
    global EMA_ALPHA_INITIAL, EMA_ALPHA_AGGREGATE_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON
    return _update_ema_z_score_and_normalize_core_with_warmup(
        scores_list, 's_ft_intermediate_z', aggregate_ema_z_score_stats,
        EMA_ALPHA_INITIAL, EMA_ALPHA_AGGREGATE_Z, EMA_INITIAL_PHASE_STEPS, EMA_EPSILON
    )

def check_answer_base(prompts, completions, ground_truths, structure_sentence, **kwargs):
    global CURRENT_EVALUATOR_KEY_FOR_EMA
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    notes = []
    print(f"Transcript:\n{question}")
    for i in range(len(responses)):
        try:
            _, note_content = responses[i].split("</think>")
            notes.append(note_content.strip())
        except:
            notes.append(None)
    failed_scores = []
    if None in notes:
        print("Note structure failed!")
        for i in range(len(notes)):
            if notes[i] == None:
                failed_scores.append(-1)
            else:
                failed_scores.append(1)
        return failed_scores, None

    base_note_from_generator = ground_truths[0]
    for i in range(len(notes)):
        print(f"Note {i}: \n{notes[i]}\n")
    print(f"Ground Base Note:\n{base_note_from_generator}")

    def evaluate_note(i_note_idx):
        if notes[i_note_idx] is not None and EVALUATORS_LIST:
            return versus(structure_sentence, question, notes[i_note_idx], base_note_from_generator, True)
        else:
            return [-4.0] * len(EVALUATORS_LIST if EVALUATORS_LIST else [1])

    with ThreadPoolExecutor() as executor:
        raw_scores_batch = list(executor.map(evaluate_note, range(len(notes))))

    if None in raw_scores_batch:
        print("Base raw scores batch failed!")
        return None, None

    mean_scores_for_wandb = []
    for single_note_raw_scores in raw_scores_batch:
        if single_note_raw_scores: mean_scores_for_wandb.append(statistics.mean(single_note_raw_scores))
        else: mean_scores_for_wandb.append(0.0)

    for i, score_val in enumerate(mean_scores_for_wandb): print(f"Score {i}:\n{score_val}")
    if mean_scores_for_wandb: print(f"\n Base Total Score (Raw Avg):\n{statistics.mean(mean_scores_for_wandb)}\n")

    s_b_intermediate = [0.0] * len(notes)
    if EVALUATORS_LIST and raw_scores_batch and len(raw_scores_batch) > 0 and raw_scores_batch[0] and len(raw_scores_batch[0]) == len(EVALUATORS_LIST):
        num_evaluators = len(EVALUATORS_LIST)
        for i_eval in range(num_evaluators):
            CURRENT_EVALUATOR_KEY_FOR_EMA = EVALUATORS_LIST[i_eval].model_name + "_BASE"
            llm_scores_one_evaluator = [raw_scores_batch[j_note][i_eval] for j_note in range(len(notes))]

            llm_scores_normalized = normalize_stage1_ema_z_score_per_evaluator(llm_scores_one_evaluator)

            for j_note in range(len(notes)):
                if j_note < len(llm_scores_normalized):
                     s_b_intermediate[j_note] += llm_scores_normalized[j_note] / num_evaluators
        CURRENT_EVALUATOR_KEY_FOR_EMA = None

    return s_b_intermediate, mean_scores_for_wandb

ft_note_first = True
def check_answer_ft(prompts, completions, ground_truths, structure_sentence, **kwargs):
    global ft_note_first
    global CURRENT_EVALUATOR_KEY_FOR_EMA
    global model
    global tokenizer
    global max_completion_length
    ft_note_first = not ft_note_first
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    notes = []
    for i in range(len(responses)):
        try:
            _, note_content = responses[i].split("</think>")
            notes.append(note_content.strip())
        except:
            notes.append(None)
    failed_scores = []
    if None in notes:
        print("Note structure failed!")
        for i in range(len(notes)):
            if notes[i] == None:
                failed_scores.append(-1)
            else:
                failed_scores.append(1)
        return failed_scores, None

    model.save_lora("./grpo_saved_lora")
    baseline_sampling_params = SamplingParams(
        temperature = 0.1,
        min_p = 0.1,
        top_p = 0.9,
        top_k = 20,
        seed = 3407,
        max_tokens = max_completion_length,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    base_note_for_ft_comparison = None
    attempts = 0
    while base_note_for_ft_comparison is None and attempts < 5:
        attempts += 1
        baseline_prompt_text = tokenizer.apply_chat_template(
            prompts[0],
            add_generation_prompt = True,
            tokenize = False,
        )
        raw_response_for_ft_comparison = model.fast_generate(
            [baseline_prompt_text],
            sampling_params = baseline_sampling_params,
            lora_request = model.load_lora("./grpo_saved_lora"),
        )[0].outputs[0].text
        shutil.rmtree("./grpo_saved_lora")
        try:
            _, base_note_for_ft_comparison = raw_response_for_ft_comparison.split("</think>")
        except:
            base_note_for_ft_comparison = None
    if base_note_for_ft_comparison is None:
        print("All Baseline Generations Failed! Returning...")
        return None, None

    num_evaluators = len(EVALUATORS_LIST)
    def evaluate_note(i):
        if notes[i] is not None: return list(versus(structure_sentence, question, notes[i], base_note_for_ft_comparison, ft_note_first))
        else: return [-4.0] * num_evaluators


    all_raw_scores_ft = []
    with ThreadPoolExecutor() as executor:
        all_raw_scores_ft = list(executor.map(evaluate_note, range(len(notes))))

    if None in all_raw_scores_ft:
        print("FT raw scores batch failed!")
        return None, None

    mean_scores_wandb_ft = []
    for rs_note in all_raw_scores_ft:
        if rs_note: mean_scores_wandb_ft.append(statistics.mean(rs_note))
        else: mean_scores_wandb_ft.append(0.0)

    for i, score_val in enumerate(mean_scores_wandb_ft): print(f"Score {i}:\n{score_val}")
    print(f"Base note for FT comparison: {base_note_for_ft_comparison}")
    print(f"\nFT Total Score (Printed Raw Avg Style):\n{statistics.mean(mean_scores_wandb_ft)}\n")

    s_ft_intermediate = [0.0] * len(notes)
    for i_eval in range(num_evaluators):
        CURRENT_EVALUATOR_KEY_FOR_EMA = EVALUATORS_LIST[i_eval].model_name + "_FT"
        llm_scores_one_evaluator = [all_raw_scores_ft[j_note][i_eval] for j_note in range(len(notes))]
        llm_scores_normalized = normalize_stage1_ema_z_score_per_evaluator(llm_scores_one_evaluator)
        for j_note in range(len(notes)):
            if j_note < len(llm_scores_normalized):
                s_ft_intermediate[j_note] += llm_scores_normalized[j_note] / num_evaluators
    CURRENT_EVALUATOR_KEY_FOR_EMA = None

    return s_ft_intermediate, mean_scores_wandb_ft

reward_weights = [1.0, 0.0]

def check_answer(prompts, completions, ground_truths, structure_sentences, **kwargs):
    global aggregate_ema_z_score_stats, per_evaluator_ema_z_score_stats
    global last_logged_optimizer_step_for_ema

    current_optimizer_step = kwargs.get('step', wandb.run.step if wandb.run else 0)

    global last_logged_optimizer_step_for_ema
    with ThreadPoolExecutor() as executor:
        future_base = executor.submit(check_answer_base, prompts, completions, ground_truths, structure_sentences[0], **kwargs)
        future_ft = executor.submit(check_answer_ft, prompts, completions, ground_truths, structure_sentences[0], **kwargs)
        s_b_intermediate, mean_scores_base_wandb = future_base.result()
        s_ft_intermediate, mean_scores_ft_wandb = future_ft.result()

    if s_b_intermediate == None or s_ft_intermediate == None:
        print("Check Answer Failed!")
        return [0.0] * len(prompts)

    log_payload = {}
    if mean_scores_ft_wandb is None:
        print(f"{s_ft_intermediate}\n")
        log_payload["generation_fails"] = s_ft_intermediate.count(-1)
        wandb.log(log_payload, step=current_optimizer_step if current_optimizer_step != -1 else None)
        return s_ft_intermediate
    log_payload["generation_fails"] = 0
    if mean_scores_base_wandb:
        log_payload["base_score_raw_avg"] = statistics.mean(mean_scores_base_wandb) if mean_scores_base_wandb else 0.0
        log_payload["base_score_raw_std"] = np.std(mean_scores_base_wandb) if len(mean_scores_base_wandb) > 1 else 0.0
    if mean_scores_ft_wandb:
        log_payload["ft_score_raw_avg"] = statistics.mean(mean_scores_ft_wandb) if mean_scores_ft_wandb else 0.0
        log_payload["ft_score_raw_std"] = np.std(mean_scores_ft_wandb) if len(mean_scores_ft_wandb) > 1 else 0.0

    final_norm_base = normalize_stage2_ema_z_score_aggregate_base(s_b_intermediate)
    final_norm_ft = normalize_stage2_ema_z_score_aggregate_ft(s_ft_intermediate)

    print(f"Scores Base (Full EMA Z-Score Normed): {final_norm_base}")
    print(f"Scores FT (Full EMA Z-Score Normed): {final_norm_ft}")

    if final_norm_base:
        log_payload["base_score_final_norm_avg"] = statistics.mean(final_norm_base)
        log_payload["base_score_final_norm_std"] = np.std(final_norm_base) if len(final_norm_base) > 1 else 0.0
    if final_norm_ft:
        log_payload["ft_score_final_norm_avg"] = statistics.mean(final_norm_ft)
        log_payload["ft_score_final_norm_std"] = np.std(final_norm_ft) if len(final_norm_ft) > 1 else 0.0

    if current_optimizer_step != -1 and current_optimizer_step != last_logged_optimizer_step_for_ema:
        for key, stats in per_evaluator_ema_z_score_stats.items():
            safe_key = str(key).replace("/", "_")
            log_payload[f"ema_stage1_z_score/{safe_key}/mean"] = stats['mean']
            log_payload[f"ema_stage1_z_score/{safe_key}/std_dev"] = math.sqrt(stats['variance'])

        for key, stats in aggregate_ema_z_score_stats.items():
            log_payload[f"ema_stage2_z_score/{key}/mean"] = stats['mean']
            log_payload[f"ema_stage2_z_score/{key}/std_dev"] = math.sqrt(stats['variance'])

        last_logged_optimizer_step_for_ema = current_optimizer_step

    if log_payload and wandb.run:
        wandb.log(log_payload, step=current_optimizer_step if current_optimizer_step != -1 else None)

    combined_scores = []
    num_rewards = len(completions)
    final_norm_base_padded = final_norm_base + [0.0] * (num_rewards - len(final_norm_base))
    final_norm_ft_padded = final_norm_ft + [0.0] * (num_rewards - len(final_norm_ft))

    for i in range(num_rewards):
        combined_score = (reward_weights[0] * final_norm_base_padded[i] +
                          reward_weights[1] * final_norm_ft_padded[i])
        combined_scores.append(combined_score)
    print(f"Combined Scores: {combined_scores}")
    return combined_scores
import wandb
import os

run_name = os.getenv("WANDB_RUN_NAME", "grpo-soap-note-full-ema-zscore")
project_name = os.getenv("WANDB_PROJECT", "AI-Scribing")

wandb.init(
    project=project_name,
    name="buck-commander-8b-unbiased",
    id="lfmf8mqr",
    reinit="finish_previous",
    resume="must",
    config={
        "ema_alpha_initial": EMA_ALPHA_INITIAL,
        "ema_initial_phase_steps": EMA_INITIAL_PHASE_STEPS,
        "ema_alpha_per_eval_z": EMA_ALPHA_PER_EVAL_Z,
        "ema_alpha_aggregate_z": EMA_ALPHA_AGGREGATE_Z,
        "reward_weights": reward_weights,
    }
)

wandb.define_metric("base_score_raw_avg", summary="mean", step_metric="train/global_step")
wandb.define_metric("base_score_raw_std", summary="mean", step_metric="train/global_step")
wandb.define_metric("ft_score_raw_avg", summary="mean", step_metric="train/global_step")
wandb.define_metric("ft_score_raw_std", summary="mean", step_metric="train/global_step")

wandb.define_metric("base_score_final_norm_avg", summary="mean", step_metric="train/global_step")
wandb.define_metric("base_score_final_norm_std", summary="mean", step_metric="train/global_step")
wandb.define_metric("ft_score_final_norm_avg", summary="mean", step_metric="train/global_step")
wandb.define_metric("ft_score_final_norm_std", summary="mean", step_metric="train/global_step")

wandb.define_metric("generation_fails", summary="mean", step_metric="train/global_step")

for evaluator in EVALUATORS_LIST:
    safe_model_name = str(evaluator.model_name).replace("/", "_")
    key_base = f"ema_stage1_z_score/{safe_model_name}"
    wandb.define_metric(f"{key_base}/mean", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base}/variance", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base}/std_dev", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base}/count", summary="last", step_metric="train/global_step")

aggregate_keys = ['s_b_intermediate_z', 's_ft_intermediate_z']
for key_agg in aggregate_keys:
    key_base_agg = f"ema_stage2_z_score/{key_agg}"
    wandb.define_metric(f"{key_base_agg}/mean", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base_agg}/variance", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base_agg}/std_dev", summary="last", step_metric="train/global_step")
    wandb.define_metric(f"{key_base_agg}/count", summary="last", step_metric="train/global_step")

print("W&B metrics defined for Full EMA Z-Score.")
maximum_length = 4000
max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 0.9,
    top_k = 20,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 0.1,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.01,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,
    num_generations = 8,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 10,
    save_steps = 4,
    report_to = "wandb",
    run_name = "buck-commander-8b-unbiased",
    output_dir = "./buck-commander-8b-unbiased",
)
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        check_answer,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train(resume_from_checkpoint="./buck-commander-8b-unbiased/checkpoint-72")
model.save_lora("lora_final_unbiased")
model.push_to_hub_merged("The-Welcomer/cluster-test-unbiased", tokenizer, save_method = "merged_16bit")
