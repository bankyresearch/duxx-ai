"""Built-in RL Environments for training LLM agents.

Environments:
    - CodingEnv: Write Python code to solve problems
    - ReasoningEnv: Math, logic, and reasoning puzzles
    - GameEnv: Simple games (number guess, 20 questions, blackjack)
    - QAEnv: Question answering with verifiable answers
    - WebSearchEnv: Find information using search tools
    - ConversationEnv: Multi-turn dialogue with goals
    - ToolUseEnv: Learn when and how to use tools correctly

Usage:
    from duxx_ai.rl.environments import CodingEnv, ReasoningEnv, GameEnv
    from duxx_ai.rl.core import rollout

    env = CodingEnv(difficulty="easy")
    episode = await rollout(agent.run, env)
"""

from __future__ import annotations

import logging
import random
import re

from duxx_ai.rl.core import Action, Observation, RLEnvironment, StepResult

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Coding Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CodingEnv(RLEnvironment):
    """Environment for training code generation agents.

    Agent receives a coding problem, writes Python code, gets test results.

    Usage:
        env = CodingEnv(difficulty="easy")
        obs = await env.reset()  # "Write a function that..."
        result = await env.step(Action(text="def solve(n): return n * 2"))
    """

    PROBLEMS = {
        "easy": [
            {"prompt": "Write a Python function `double(n)` that returns n * 2.", "test": "assert double(5) == 10 and double(0) == 0 and double(-3) == -6", "solution_hint": "def double(n): return n * 2"},
            {"prompt": "Write a Python function `is_even(n)` that returns True if n is even.", "test": "assert is_even(4) == True and is_even(7) == False and is_even(0) == True", "solution_hint": "def is_even(n): return n % 2 == 0"},
            {"prompt": "Write a Python function `reverse_string(s)` that reverses a string.", "test": "assert reverse_string('hello') == 'olleh' and reverse_string('') == '' and reverse_string('a') == 'a'", "solution_hint": "def reverse_string(s): return s[::-1]"},
            {"prompt": "Write a Python function `max_of_three(a, b, c)` that returns the largest.", "test": "assert max_of_three(1,2,3) == 3 and max_of_three(5,1,2) == 5 and max_of_three(-1,-2,-3) == -1", "solution_hint": "def max_of_three(a,b,c): return max(a,b,c)"},
            {"prompt": "Write a Python function `count_vowels(s)` that counts vowels in a string.", "test": "assert count_vowels('hello') == 2 and count_vowels('xyz') == 0 and count_vowels('aeiou') == 5", "solution_hint": "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')"},
        ],
        "medium": [
            {"prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.", "test": "assert fibonacci(0) == 0 and fibonacci(1) == 1 and fibonacci(10) == 55", "solution_hint": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a"},
            {"prompt": "Write a Python function `flatten(lst)` that flattens a nested list.", "test": "assert flatten([1,[2,[3,4],5],6]) == [1,2,3,4,5,6] and flatten([]) == []", "solution_hint": "def flatten(lst):\n    r = []\n    for i in lst:\n        if isinstance(i, list): r.extend(flatten(i))\n        else: r.append(i)\n    return r"},
            {"prompt": "Write a Python function `is_palindrome(s)` that checks if a string is a palindrome (ignoring case and spaces).", "test": "assert is_palindrome('racecar') == True and is_palindrome('A man a plan a canal Panama') == True and is_palindrome('hello') == False", "solution_hint": "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]"},
            {"prompt": "Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add up to target.", "test": "r = two_sum([2,7,11,15], 9); assert sorted(r) == [0,1]", "solution_hint": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen: return [seen[target-n], i]\n        seen[n] = i"},
        ],
        "hard": [
            {"prompt": "Write a Python function `lru_cache(capacity)` that returns a class with get(key) and put(key, value) methods. get returns -1 if not found.", "test": "c = lru_cache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1", "solution_hint": "from collections import OrderedDict\ndef lru_cache(cap):\n    class LRU:\n        def __init__(self): self.d = OrderedDict(); self.cap = cap\n        def get(self, k):\n            if k not in self.d: return -1\n            self.d.move_to_end(k); return self.d[k]\n        def put(self, k, v):\n            if k in self.d: self.d.move_to_end(k)\n            self.d[k] = v\n            if len(self.d) > self.cap: self.d.popitem(last=False)\n    return LRU()"},
        ],
    }

    def __init__(self, difficulty: str = "easy", **kwargs):
        super().__init__(name="coding", max_steps=5, **kwargs)
        self.difficulty = difficulty
        self._problem: dict = {}
        self._attempts = 0

    async def reset(self, **kwargs) -> Observation:
        problems = self.PROBLEMS.get(self.difficulty, self.PROBLEMS["easy"])
        self._problem = random.choice(problems)
        self._attempts = 0
        return Observation(
            text=f"{self._problem['prompt']}\n\nWrite ONLY the Python code. No explanations.",
            data={"difficulty": self.difficulty},
        )

    async def step(self, action: Action) -> StepResult:
        self._attempts += 1
        code = action.text.strip()
        # Remove markdown code fences if present
        code = re.sub(r'^```python\s*\n?', '', code)
        code = re.sub(r'\n?```\s*$', '', code)

        try:
            # Execute code + test
            namespace: dict = {}
            exec(code, namespace)
            exec(self._problem["test"], namespace)
            # All tests passed
            return StepResult(
                observation=Observation(text="All tests passed! Correct solution."),
                reward=1.0,
                done=True,
                info={"correct": True, "attempts": self._attempts},
            )
        except AssertionError:
            return StepResult(
                observation=Observation(text=f"Tests failed. Your code runs but produces wrong output. Try again.\nTests: {self._problem['test']}"),
                reward=-0.1,
                info={"correct": False, "error": "assertion"},
            )
        except Exception as e:
            return StepResult(
                observation=Observation(text=f"Code error: {type(e).__name__}: {e}\n\nFix your code and try again."),
                reward=-0.2,
                info={"correct": False, "error": str(e)},
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reasoning Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReasoningEnv(RLEnvironment):
    """Environment for training reasoning and math agents.

    Agent receives math/logic problems, gives answers, gets feedback.

    Usage:
        env = ReasoningEnv(category="math")
        obs = await env.reset()
        result = await env.step(Action(text="42"))
    """

    PROBLEMS = {
        "math": [
            {"question": "What is 17 * 23?", "answer": "391"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "If x + 5 = 13, what is x?", "answer": "8"},
            {"question": "What is 2^10?", "answer": "1024"},
            {"question": "What is 15% of 200?", "answer": "30"},
            {"question": "What is the sum of first 10 positive integers?", "answer": "55"},
            {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": "40"},
            {"question": "What is the GCD of 24 and 36?", "answer": "12"},
            {"question": "If a train travels 120 km in 2 hours, what is its speed in km/h?", "answer": "60"},
            {"question": "What is 3! + 4! ?", "answer": "30"},
        ],
        "logic": [
            {"question": "If all roses are flowers, and all flowers are plants, are all roses plants? Answer yes or no.", "answer": "yes"},
            {"question": "I have 3 boxes. Box A is heavier than Box B. Box C is lighter than Box B. Which box is heaviest?", "answer": "a"},
            {"question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "answer": "5"},
            {"question": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have?", "answer": "9"},
            {"question": "If you have a bowl with 6 apples and you take away 4, how many do you have?", "answer": "4"},
        ],
        "word": [
            {"question": "What word becomes shorter when you add two letters to it?", "answer": "short"},
            {"question": "I speak without a mouth and hear without ears. I have nobody, but come alive with the wind. What am I?", "answer": "echo"},
            {"question": "What has keys but no locks, space but no room, and you can enter but can't go inside?", "answer": "keyboard"},
        ],
    }

    def __init__(self, category: str = "math", **kwargs):
        super().__init__(name="reasoning", max_steps=3, **kwargs)
        self.category = category
        self._problem: dict = {}

    async def reset(self, **kwargs) -> Observation:
        problems = self.PROBLEMS.get(self.category, self.PROBLEMS["math"])
        self._problem = random.choice(problems)
        return Observation(
            text=f"Solve this problem. Give ONLY the answer, no explanation.\n\n{self._problem['question']}",
            data={"category": self.category},
        )

    async def step(self, action: Action) -> StepResult:
        answer = action.text.strip().lower().rstrip(".")
        expected = self._problem["answer"].lower()

        # Check if answer matches (flexible matching)
        correct = (
            answer == expected
            or answer.replace(",", "") == expected
            or expected in answer.split()
            or (answer.replace(" ", "") == expected.replace(" ", ""))
        )

        if correct:
            return StepResult(
                observation=Observation(text=f"Correct! The answer is {expected}."),
                reward=1.0, done=True,
                info={"correct": True},
            )
        else:
            return StepResult(
                observation=Observation(text=f"Incorrect. Try again. The question was: {self._problem['question']}"),
                reward=-0.1,
                info={"correct": False, "your_answer": answer, "expected": expected},
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Game Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NumberGuessEnv(RLEnvironment):
    """Guess a number between 1 and N. Agent gets higher/lower hints."""

    def __init__(self, max_number: int = 100, **kwargs):
        super().__init__(name="number_guess", max_steps=10, **kwargs)
        self.max_number = max_number
        self._target = 0

    async def reset(self, **kwargs) -> Observation:
        self._target = random.randint(1, self.max_number)
        return Observation(text=f"I'm thinking of a number between 1 and {self.max_number}. Guess it! Reply with just a number.")

    async def step(self, action: Action) -> StepResult:
        try:
            guess = int(re.search(r'\d+', action.text).group())
        except:
            return StepResult(observation=Observation(text="Please respond with a number."), reward=-0.1)

        diff = abs(guess - self._target)
        if diff == 0:
            return StepResult(observation=Observation(text=f"Correct! It was {self._target}!"), reward=1.0, done=True, info={"correct": True})
        elif guess < self._target:
            return StepResult(observation=Observation(text=f"{guess} is too low. Try higher."), reward=-0.01 * diff / self.max_number)
        else:
            return StepResult(observation=Observation(text=f"{guess} is too high. Try lower."), reward=-0.01 * diff / self.max_number)


class BlackjackEnv(RLEnvironment):
    """Simple Blackjack environment. Agent decides hit or stand."""

    def __init__(self, **kwargs):
        super().__init__(name="blackjack", max_steps=10, **kwargs)
        self._hand: list[int] = []
        self._dealer: list[int] = []

    def _draw(self) -> int:
        card = random.randint(1, 13)
        return min(card, 10)  # Face cards = 10

    def _hand_value(self, hand: list[int]) -> int:
        total = sum(hand)
        # Ace handling: if we have an ace (1) and can use it as 11
        if 1 in hand and total + 10 <= 21:
            total += 10
        return total

    async def reset(self, **kwargs) -> Observation:
        self._hand = [self._draw(), self._draw()]
        self._dealer = [self._draw(), self._draw()]
        val = self._hand_value(self._hand)
        return Observation(
            text=f"Your hand: {self._hand} (value: {val}). Dealer shows: {self._dealer[0]}. Say 'hit' or 'stand'.",
            data={"hand": self._hand, "hand_value": val, "dealer_showing": self._dealer[0]},
        )

    async def step(self, action: Action) -> StepResult:
        choice = action.text.strip().lower()

        if "hit" in choice:
            self._hand.append(self._draw())
            val = self._hand_value(self._hand)
            if val > 21:
                return StepResult(observation=Observation(text=f"Bust! Your hand: {self._hand} (value: {val})."), reward=-1.0, done=True, info={"result": "bust"})
            return StepResult(observation=Observation(text=f"Your hand: {self._hand} (value: {val}). Dealer shows: {self._dealer[0]}. Hit or stand?"), reward=0.0)

        elif "stand" in choice:
            # Dealer plays
            while self._hand_value(self._dealer) < 17:
                self._dealer.append(self._draw())
            player_val = self._hand_value(self._hand)
            dealer_val = self._hand_value(self._dealer)

            if dealer_val > 21:
                return StepResult(observation=Observation(text=f"Dealer busts! Dealer: {self._dealer} ({dealer_val}). You win!"), reward=1.0, done=True, info={"result": "win"})
            elif player_val > dealer_val:
                return StepResult(observation=Observation(text=f"You win! You: {player_val}, Dealer: {dealer_val}."), reward=1.0, done=True, info={"result": "win"})
            elif player_val == dealer_val:
                return StepResult(observation=Observation(text=f"Push! Both have {player_val}."), reward=0.0, done=True, info={"result": "push"})
            else:
                return StepResult(observation=Observation(text=f"Dealer wins. You: {player_val}, Dealer: {dealer_val}."), reward=-1.0, done=True, info={"result": "lose"})

        return StepResult(observation=Observation(text="Say 'hit' or 'stand'."), reward=-0.05)


class MazeEnv(RLEnvironment):
    """Simple text-based maze navigation."""

    def __init__(self, size: int = 5, **kwargs):
        super().__init__(name="maze", max_steps=size * size, **kwargs)
        self.size = size
        self._pos = [0, 0]
        self._goal = [0, 0]
        self._walls: set = set()

    async def reset(self, **kwargs) -> Observation:
        self._pos = [0, 0]
        self._goal = [self.size - 1, self.size - 1]
        # Random walls (20% of cells, not start/end)
        self._walls = set()
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in [(0, 0), tuple(self._goal)] and random.random() < 0.2:
                    self._walls.add((r, c))
        return Observation(
            text=f"Navigate a {self.size}x{self.size} maze. You are at (0,0). Goal is at ({self._goal[0]},{self._goal[1]}). Commands: up, down, left, right.\nWalls at: {sorted(self._walls)[:5]}{'...' if len(self._walls) > 5 else ''}",
            data={"position": self._pos.copy(), "goal": self._goal, "size": self.size},
        )

    async def step(self, action: Action) -> StepResult:
        direction = action.text.strip().lower()
        r, c = self._pos
        if "up" in direction: r -= 1
        elif "down" in direction: r += 1
        elif "left" in direction: c -= 1
        elif "right" in direction: c += 1
        else:
            return StepResult(observation=Observation(text=f"Invalid. Use up/down/left/right. You're at ({self._pos[0]},{self._pos[1]})."), reward=-0.05)

        # Bounds check
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return StepResult(observation=Observation(text=f"Wall! Can't go there. You're at ({self._pos[0]},{self._pos[1]})."), reward=-0.1)
        # Wall check
        if (r, c) in self._walls:
            return StepResult(observation=Observation(text=f"Wall at ({r},{c})! You're at ({self._pos[0]},{self._pos[1]})."), reward=-0.1)

        self._pos = [r, c]
        dist = abs(r - self._goal[0]) + abs(c - self._goal[1])

        if self._pos == self._goal:
            return StepResult(observation=Observation(text="You reached the goal!"), reward=1.0, done=True, info={"correct": True})

        return StepResult(
            observation=Observation(text=f"Moved to ({r},{c}). Distance to goal: {dist}. Commands: up/down/left/right."),
            reward=-0.01,  # Small step penalty
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QA Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QAEnv(RLEnvironment):
    """Question-answering environment with verifiable answers."""

    QUESTIONS = [
        {"q": "What is the capital of France?", "a": "paris"},
        {"q": "What planet is closest to the sun?", "a": "mercury"},
        {"q": "What is the chemical symbol for water?", "a": "h2o"},
        {"q": "How many sides does a hexagon have?", "a": "6"},
        {"q": "What year did World War II end?", "a": "1945"},
        {"q": "What is the largest ocean on Earth?", "a": "pacific"},
        {"q": "What element has atomic number 1?", "a": "hydrogen"},
        {"q": "How many continents are there?", "a": "7"},
        {"q": "What is the speed of light in km/s (approximately)?", "a": "300000"},
        {"q": "Who wrote Romeo and Juliet?", "a": "shakespeare"},
    ]

    def __init__(self, **kwargs):
        super().__init__(name="qa", max_steps=2, **kwargs)
        self._qa: dict = {}

    async def reset(self, **kwargs) -> Observation:
        self._qa = random.choice(self.QUESTIONS)
        return Observation(text=f"Answer this question with just the answer, no explanation:\n{self._qa['q']}")

    async def step(self, action: Action) -> StepResult:
        answer = action.text.strip().lower().rstrip(".")
        expected = self._qa["a"]
        correct = expected in answer or answer == expected
        if correct:
            return StepResult(observation=Observation(text="Correct!"), reward=1.0, done=True, info={"correct": True})
        return StepResult(
            observation=Observation(text=f"Incorrect. The answer is: {expected}"),
            reward=-0.5, done=True, info={"correct": False, "expected": expected},
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Use Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ToolUseEnv(RLEnvironment):
    """Train agents to use tools correctly. Agent must decide which tool to use."""

    TASKS = [
        {"task": "Calculate 847 * 293", "correct_tool": "calculator", "answer": "248171"},
        {"task": "Read the file config.json", "correct_tool": "read_file", "answer": "read_file"},
        {"task": "Search the web for 'Python 3.12 release date'", "correct_tool": "web_request", "answer": "web_request"},
        {"task": "List all files in the current directory", "correct_tool": "list_files", "answer": "list_files"},
        {"task": "Run a Python script that prints Hello World", "correct_tool": "python_exec", "answer": "python_exec"},
    ]

    def __init__(self, **kwargs):
        super().__init__(name="tool_use", max_steps=3, **kwargs)
        self._task: dict = {}

    async def reset(self, **kwargs) -> Observation:
        self._task = random.choice(self.TASKS)
        return Observation(
            text=f"You have these tools: calculator, read_file, web_request, list_files, python_exec.\n\nTask: {self._task['task']}\n\nWhich tool should you use? Reply with just the tool name.",
            data={"available_tools": ["calculator", "read_file", "web_request", "list_files", "python_exec"]},
        )

    async def step(self, action: Action) -> StepResult:
        tool = action.text.strip().lower().replace(" ", "_")
        correct = self._task["correct_tool"]
        if correct in tool or tool in correct:
            return StepResult(observation=Observation(text=f"Correct! {correct} is the right tool."), reward=1.0, done=True, info={"correct": True})
        return StepResult(
            observation=Observation(text=f"Not quite. The correct tool for '{self._task['task']}' is {correct}."),
            reward=-0.5, done=True, info={"correct": False, "expected": correct},
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Conversation Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConversationEnv(RLEnvironment):
    """Multi-turn conversation with a goal. Agent must achieve the conversational objective."""

    SCENARIOS = [
        {"goal": "Get the user's name and email address", "persona": "I'm a new customer looking for help.", "success_keys": ["name", "email"]},
        {"goal": "Help the user troubleshoot a login issue", "persona": "I can't log into my account. I keep getting 'invalid password'.", "success_keys": ["reset", "password"]},
        {"goal": "Convince the user to upgrade to the premium plan", "persona": "I'm happy with the free plan. Why would I upgrade?", "success_keys": ["upgrade", "premium", "benefits"]},
    ]

    def __init__(self, **kwargs):
        super().__init__(name="conversation", max_steps=10, **kwargs)
        self._scenario: dict = {}
        self._turn = 0

    async def reset(self, **kwargs) -> Observation:
        self._scenario = random.choice(self.SCENARIOS)
        self._turn = 0
        return Observation(
            text=f"GOAL: {self._scenario['goal']}\n\nUser: {self._scenario['persona']}",
            data={"goal": self._scenario["goal"]},
        )

    async def step(self, action: Action) -> StepResult:
        self._turn += 1
        response = action.text.lower()

        # Check if agent addressed the goal keywords
        hits = sum(1 for k in self._scenario["success_keys"] if k in response)
        total = len(self._scenario["success_keys"])

        if hits == total:
            return StepResult(
                observation=Observation(text="Great, that helps! Thank you."),
                reward=1.0, done=True, info={"correct": True, "turns": self._turn},
            )
        elif hits > 0:
            return StepResult(
                observation=Observation(text="User: That's helpful, but I need more information..."),
                reward=0.1 * hits,
            )
        else:
            return StepResult(
                observation=Observation(text=f"User: Hmm, that doesn't really help. {self._scenario['persona']}"),
                reward=-0.05,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Environment Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENVIRONMENTS: dict[str, type[RLEnvironment]] = {
    "coding": CodingEnv,
    "reasoning": ReasoningEnv,
    "number_guess": NumberGuessEnv,
    "blackjack": BlackjackEnv,
    "maze": MazeEnv,
    "qa": QAEnv,
    "tool_use": ToolUseEnv,
    "conversation": ConversationEnv,
}


def list_environments() -> list[dict[str, str]]:
    """List all available environments."""
    return [{"name": name, "class": cls.__name__, "doc": (cls.__doc__ or "").split("\n")[0]} for name, cls in ENVIRONMENTS.items()]


def create_environment(name: str, **kwargs) -> RLEnvironment:
    """Create an environment by name."""
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENVIRONMENTS.keys())}")
    return ENVIRONMENTS[name](**kwargs)
