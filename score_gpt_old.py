"""
EECS 595 HW3: GPT Model Evaluation Script

This script evaluates the SFT-trained GPT model on multiple choice questions
from test_questions.jsonl and generates detailed performance reports.

Usage:
    python score_gpt.py

The script will:
1. Load the SFT-trained model
2. Process each question from test_questions.jsonl
3. Generate responses using the model
4. Parse answers using both strict and loose parsing
5. Generate a CSV report with detailed results
6. Print accuracy scores
"""

import os
import json
import csv
import re
import torch
import sft
import gpt
from transformers import AutoTokenizer
import warnings
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate SFT-trained GPT model on multiple choice questions')

    # Model arguments
    parser.add_argument('--model_path', type=str,
                       default='models/sft-models/sft-gpt-1000-step.pth',
                       help='Path to SFT-trained model checkpoint')
    parser.add_argument('--questions_file', type=str,
                       default='test_questions.jsonl',
                       help='Path to questions file')
    parser.add_argument('--output_file', type=str,
                       default=f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                       help='Output CSV file path')

    # Model configuration
    parser.add_argument('--vocab_size', type=int, default=50262,
                       help='Model vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Generation parameters
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')

    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load the SFT-trained model and tokenizer."""
    print("🔄 Loading tokenizer...")
    tokenizer = gpt.setup_tokenizer()
    print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.vocab_size}")

    print("🔄 Loading model...")
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
    }

    model = sft.load_pretrained_model(args.model_path, model_config)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    model = model.to(device)
    print(f"✅ Model loaded and moved to {device}!")

    return model, tokenizer, device


def load_questions(questions_file: str) -> List[Dict]:
    """Load questions from JSONL file."""
    print(f"🔄 Loading questions from {questions_file}...")

    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file {questions_file} not found!")

    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                question_data = json.loads(line)
                questions.append(question_data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue

    print(f"✅ Loaded {len(questions)} questions")
    return questions


def format_question_prompt(question_data: Dict) -> str:
    """Format a question as a prompt for the model."""
    question = question_data['question']
    options = question_data['options']

    # Create the prompt
    prompt = f"Question: {question}\n\n"
    prompt += "Please choose the correct answer from the following options:\n"

    for option, text in options.items():
        prompt += f"{option}. {text}\n"

    prompt += "\nGenerate only the letter of the correct answer.\nAnswer:"

    return prompt


def generate_model_response(model, tokenizer, prompt: str, device: str,
                          max_tokens: int, temperature: float) -> str:
    """Generate a response from the model."""
    model.eval()

    with torch.no_grad():
        # Format the prompt with special tokens
        formatted_input = f"<|user|>{prompt}<|end|><|assistant|>"

        # Tokenize input
        input_ids = tokenizer.encode(formatted_input, return_tensors="pt")
        input_ids = input_ids.to(device)

        # Generate response
        generated_ids = input_ids.clone()

        for _ in range(max_tokens):
            # Get model output
            with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                logits = model(generated_ids)

            # Get logits for the last token
            next_token_logits = logits[0, -1, :] / temperature

            # Apply softmax and sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if we hit the end token
            if next_token.item() == tokenizer.convert_tokens_to_ids("<|end|>"):
                break

            # Stop if we hit max context length
            if generated_ids.shape[1] >= model.context_length:
                break

        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # Extract just the assistant's response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1]
            if "<|end|>" in response:
                response = response.split("<|end|>")[0]
            return response.strip()
        else:
            return "No valid response generated."


def parse_answer_strict(response: str) -> Optional[str]:
    """
    Strict parsing: Look for exact option letters (A, B, C, D) in the response.
    """
    # Look for standalone option letters
    pattern = r'\b([ABCD])\b'
    matches = re.findall(pattern, response.upper())

    if matches:
        return matches[0]  # Return the first match

    return None


def parse_answer_loose(response: str) -> Optional[str]:
    """
    Loose parsing: Look for option letters in various contexts.
    """
    response_upper = response.upper()

    # Look for patterns like "Answer: A", "The answer is B", "Option C", etc.
    patterns = [
        r'(?:answer|option|choice)[\s:]*([ABCD])',
        r'([ABCD])[\s]*[\.\)]',  # A. or A)
        r'\b([ABCD])\b',  # Standalone letters
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_upper)
        if matches:
            return matches[0]

    return None


def evaluate_model(model, tokenizer, device: str, questions: List[Dict],
                  max_tokens: int, temperature: float) -> List[Dict]:
    """Evaluate the model on all questions."""
    print(f"🔄 Evaluating model on {len(questions)} questions...")

    results = []

    for i, question_data in enumerate(questions):
        print(f"📝 Processing question {i+1}/{len(questions)}: {question_data['question'][:50]}...")

        # Format the question prompt
        prompt = format_question_prompt(question_data)

        # Generate model response
        try:
            response = generate_model_response(
                model, tokenizer, prompt, device, max_tokens, temperature
            )
        except Exception as e:
            print(f"⚠️  Error generating response for question {i+1}: {e}")
            response = f"Error: {str(e)}"

        # Parse answers
        strict_answer = parse_answer_strict(response)
        loose_answer = parse_answer_loose(response)

        # Check correctness
        correct_answer = question_data['answer']
        strict_correct = (strict_answer == correct_answer) if strict_answer else False
        loose_correct = (loose_answer == correct_answer) if loose_answer else False

        # Store results
        result = {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'options': json.dumps(question_data['options']),
            'correct_answer': correct_answer,
            'model_response': response,
            'strict_parsed_answer': strict_answer,
            'loose_parsed_answer': loose_answer,
            'strict_correct': strict_correct,
            'loose_correct': loose_correct,
            'topic': question_data['metadata']['topic'],
            'difficulty': question_data['metadata']['difficulty']
        }

        results.append(result)

        # Print progress
        status = "✅" if (strict_correct or loose_correct) else "❌"
        response_preview = response[:50] + "..." if len(response) > 50 else response
        print(f"   {status} Strict: {strict_answer}, Loose: {loose_answer}, Correct: {correct_answer}")
        print(f"   Response: {response_preview}")

    print("✅ Evaluation completed!")
    return results


def save_results_to_csv(results: List[Dict], output_file: str):
    """Save evaluation results to CSV file."""
    print(f"💾 Saving results to {output_file}...")

    if not results:
        print("⚠️  No results to save!")
        return

    fieldnames = [
        'question_id', 'question', 'options', 'correct_answer',
        'model_response', 'strict_parsed_answer', 'loose_parsed_answer',
        'strict_correct', 'loose_correct', 'topic', 'difficulty'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Results saved to {output_file}")


def calculate_and_print_scores(results: List[Dict]):
    """Calculate and print detailed accuracy scores."""
    if not results:
        print("⚠️  No results to analyze!")
        return

    total_questions = len(results)
    strict_correct = sum(1 for r in results if r['strict_correct'])
    loose_correct = sum(1 for r in results if r['loose_correct'])

    strict_accuracy = strict_correct / total_questions * 100
    loose_accuracy = loose_correct / total_questions * 100

    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total Questions: {total_questions}")
    print(f"Strict Parsing Accuracy: {strict_correct}/{total_questions} ({strict_accuracy:.2f}%)")
    print(f"Loose Parsing Accuracy: {loose_correct}/{total_questions} ({loose_accuracy:.2f}%)")

    # Analyze by difficulty
    print("\n📈 Results by Difficulty:")
    difficulties = {}
    for result in results:
        diff = result['difficulty']
        if diff not in difficulties:
            difficulties[diff] = {'total': 0, 'strict_correct': 0, 'loose_correct': 0}

        difficulties[diff]['total'] += 1
        if result['strict_correct']:
            difficulties[diff]['strict_correct'] += 1
        if result['loose_correct']:
            difficulties[diff]['loose_correct'] += 1

    for diff, stats in difficulties.items():
        strict_acc = stats['strict_correct'] / stats['total'] * 100
        loose_acc = stats['loose_correct'] / stats['total'] * 100
        print(f"  {diff.title()}: {stats['strict_correct']}/{stats['total']} strict ({strict_acc:.1f}%), "
              f"{stats['loose_correct']}/{stats['total']} loose ({loose_acc:.1f}%)")

    # Analyze by topic
    print("\n📚 Results by Topic:")
    topics = {}
    for result in results:
        topic = result['topic']
        if topic not in topics:
            topics[topic] = {'total': 0, 'strict_correct': 0, 'loose_correct': 0}

        topics[topic]['total'] += 1
        if result['strict_correct']:
            topics[topic]['strict_correct'] += 1
        if result['loose_correct']:
            topics[topic]['loose_correct'] += 1

    # Sort topics by total questions
    sorted_topics = sorted(topics.items(), key=lambda x: x[1]['total'], reverse=True)

    for topic, stats in sorted_topics[:10]:  # Show top 10 topics
        strict_acc = stats['strict_correct'] / stats['total'] * 100
        loose_acc = stats['loose_correct'] / stats['total'] * 100
        print(f"  {topic.title()}: {stats['strict_correct']}/{stats['total']} strict ({strict_acc:.1f}%), "
              f"{stats['loose_correct']}/{stats['total']} loose ({loose_acc:.1f}%)")

    print("="*60)


def main():
    """Main evaluation function."""
    print("🚀 Starting GPT Model Evaluation")
    print("="*50)

    # Parse arguments
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args)

    # Load questions
    questions = load_questions(args.questions_file)

    # Evaluate model
    results = evaluate_model(
        model, tokenizer, device, questions,
        args.max_tokens, args.temperature
    )

    # Save results
    save_results_to_csv(results, args.output_file)

    # Calculate and print scores
    calculate_and_print_scores(results)

    print(f"\n🎉 Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
