"""LLM-as-judge scoring: two judges from different providers score each response 1-5 on instruction following,
helpfulness and fluency. API keys come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY).

Judge model names are set by the two environment variables below so the script keeps working as models change.
"""
import json
import os
import time

import anthropic
import pandas as pd
from openai import OpenAI

JUDGE_A_MODEL = os.environ.get("JUDGE_A_MODEL", "claude-sonnet-4-6")  # Anthropic API model id
JUDGE_B_MODEL = os.environ.get("JUDGE_B_MODEL", "gpt-5.4")  # OpenAI API model id

SYSTEM = """You are an expert evaluator for language model outputs.
Score responses strictly on three dimensions (1=worst, 5=best):
- instruction_following: Does the response directly fulfill the instruction?
- helpfulness: Is the response practically useful and complete?
- fluency: Is the response grammatically correct and coherent?
Reply ONLY with valid JSON. No explanation, no markdown."""
USER = 'Instruction: {instruction}\nResponse: {response}\n{{"instruction_following": <1-5>, "helpfulness": <1-5>, "fluency": <1-5>}}'

a_client, b_client = anthropic.Anthropic(), OpenAI()


def parse(text):
    try:
        d = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return {k: max(1, min(5, int(d[k]))) for k in ("instruction_following", "helpfulness", "fluency")}
    except Exception:
        return None  # unlike the original notebook, a parse failure is NOT silently scored 3


def retry(fn):
    for attempt in range(3):
        try:
            return parse(fn())
        except Exception:
            time.sleep(30)
    return None


def judge_a(instruction, response):
    return retry(lambda: a_client.messages.create(
        model=JUDGE_A_MODEL, max_tokens=100, system=SYSTEM,
        messages=[{"role": "user", "content": USER.format(instruction=instruction[:300], response=response[:400])}],
    ).content[0].text)


def judge_b(instruction, response):
    return retry(lambda: b_client.chat.completions.create(
        model=JUDGE_B_MODEL, max_completion_tokens=100,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": USER.format(instruction=instruction[:300], response=response[:400])}],
    ).choices[0].message.content)


if __name__ == "__main__":
    df = pd.read_csv("part1_responses.csv")
    for col in ("ft_response",):
        for i, r in df.iterrows():
            for tag, fn in (("a", judge_a), ("b", judge_b)):
                s = fn(r["instruction"], str(r[col])) or {}
                for k in ("instruction_following", "helpfulness", "fluency"):
                    df.at[i, f"judge_{tag}_{k}"] = s.get(k)
    df.to_csv("part1_judged.csv", index=False)
