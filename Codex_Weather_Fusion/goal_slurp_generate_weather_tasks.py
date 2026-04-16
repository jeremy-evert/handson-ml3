import json
import subprocess
from pathlib import Path

GOALS_FILE = "weather_fusion_goals.json"
TASKS_FILE = "weather_fusion_tasks.json"

MAX_TASKS_PER_GOAL = 1


def load_goals():
    with open(GOALS_FILE, "r") as f:
        return json.load(f)


def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump({"tasks": tasks}, f, indent=2)


def run_codex(prompt):

    result = subprocess.run(
        ["codex", "exec", prompt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def build_prompt(goal):

    return f"""
You are helping design software tasks.

GOAL
-----
{goal['title']}

DESCRIPTION
-----------
{goal['description']}

INSTRUCTIONS
-------------
Break this goal into small implementation tasks.

Each task should:
- be achievable in a single coding step
- represent one function or one small feature
- follow clean software engineering practices

Limit the result to {MAX_TASKS_PER_GOAL} tasks.

Return JSON in the format:

[
  {{
    "title": "task title",
    "description": "task description"
  }}
]

Return only JSON.
"""


def generate_tasks(goals_data):

    all_tasks = []
    task_counter = 1

    for goal in goals_data["goals"]:

        print("\n--------------------------------")
        print("Processing Goal:", goal["id"])
        print(goal["title"])
        print("--------------------------------")

        prompt = build_prompt(goal)

        response = run_codex(prompt)

        try:
            tasks = json.loads(response)
        except:
            print("Failed to parse Codex output")
            print(response)
            continue

        for task in tasks:

            task_entry = {
                "id": f"T-{task_counter:03}",
                "goal": goal["id"],
                "title": task["title"],
                "description": task["description"],
                "status": "todo"
            }

            all_tasks.append(task_entry)
            task_counter += 1

    return all_tasks


def main():

    print("\nWeather Fusion Goal → Task Generator\n")

    goals_data = load_goals()

    tasks = generate_tasks(goals_data)

    save_tasks(tasks)

    print("\nGenerated", len(tasks), "tasks")
    print("Saved to", TASKS_FILE)


if __name__ == "__main__":
    main()
