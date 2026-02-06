from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tinycodetest.schema import Task, TestCase

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

DifficultyTemplate = Callable[[random.Random, str], Task]


def _build_prompt(signature: str, instructions: str) -> str:
    return (
        "You are given a Python coding task.\\n"
        "Write only valid Python code implementing the function below.\\n\\n"
        f"Signature:\\n{signature}\\n\\n"
        f"Task:\\n{instructions}"
    )


def _rand_word(rng: random.Random, min_len: int = 3, max_len: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    size = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(size))


def _canon(solution: str) -> str:
    return solution.strip() + "\n"


def _task(
    *,
    task_id: str,
    difficulty: str,
    signature: str,
    function_name: str,
    instructions: str,
    canonical_solution: str,
    test_cases: list[TestCase],
    tags: list[str],
    adversarial: bool,
    template_name: str,
) -> Task:
    return Task(
        task_id=task_id,
        difficulty=difficulty,
        prompt=_build_prompt(signature, instructions),
        function_name=function_name,
        signature=signature,
        canonical_solution=_canon(canonical_solution),
        test_cases=test_cases,
        tags=tags,
        adversarial=adversarial,
        metadata={
            "template": template_name,
            "bucket": difficulty,
            "split": "adversarial" if adversarial else "main",
        },
    )


def _easy_add_constant(rng: random.Random, task_id: str) -> Task:
    fn = f"add_constant_{task_id.lower().replace('-', '_')}"
    delta = rng.choice([x for x in range(-40, 41) if x != 0])
    signature = f"def {fn}(x: int) -> int:"
    instructions = (
        f"Return x shifted by a fixed constant of {delta}. "
        "Do not mutate inputs and do not print anything."
    )
    cases = [-100, -7, -1, 0, 1, 19, 87, rng.randint(-500, 500)]
    test_cases = [TestCase(args=[x], kwargs={}, expected=x + delta) for x in cases]
    code = f"""
def {fn}(x: int) -> int:
    return x + ({delta})
"""
    return _task(
        task_id=task_id,
        difficulty=EASY,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["arithmetic", "offset", "easy"],
        adversarial=delta < 0,
        template_name="add_constant",
    )


def _easy_count_char(rng: random.Random, task_id: str) -> Task:
    fn = f"count_char_{task_id.lower().replace('-', '_')}"
    target = rng.choice(list("abcdefghijklmnopqrstuvwxyz -_"))
    signature = f"def {fn}(text: str) -> int:"
    instructions = (
        f"Return the number of occurrences of the character {target!r} in text. "
        "The comparison is case-sensitive."
    )
    strings = [
        "",
        "banana",
        "AlphaBeta",
        "space space",
        "__init__",
        "abracadabra",
        "-a-a-a-",
        " ",
    ]
    rng.shuffle(strings)
    test_cases = [TestCase(args=[s], kwargs={}, expected=s.count(target)) for s in strings]
    code = f"""
def {fn}(text: str) -> int:
    return text.count({target!r})
"""
    return _task(
        task_id=task_id,
        difficulty=EASY,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["strings", "count", "easy"],
        adversarial=target in {" ", "-", "_"},
        template_name="count_char",
    )


def _easy_clamp(rng: random.Random, task_id: str) -> Task:
    fn = f"clamp_value_{task_id.lower().replace('-', '_')}"
    lo = rng.randint(-25, 5)
    hi = lo + rng.randint(4, 25)
    signature = f"def {fn}(x: int) -> int:"
    instructions = f"Clamp x into the inclusive range [{lo}, {hi}]."

    def clamp(x: int) -> int:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    values = [lo - 100, lo - 1, lo, lo + 1, hi - 1, hi, hi + 1, hi + 100, rng.randint(-50, 60)]
    test_cases = [TestCase(args=[x], kwargs={}, expected=clamp(x)) for x in values]
    code = f"""
def {fn}(x: int) -> int:
    if x < {lo}:
        return {lo}
    if x > {hi}:
        return {hi}
    return x
"""
    return _task(
        task_id=task_id,
        difficulty=EASY,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["arithmetic", "bounds", "easy"],
        adversarial=True,
        template_name="clamp",
    )


def _easy_reverse_words(_: random.Random, task_id: str) -> Task:
    fn = f"reverse_words_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(text: str) -> str:"
    instructions = (
        "Return the words in reverse order. "
        "Input may contain repeated spaces, and output must contain single spaces only."
    )
    strings = [
        "hello world",
        "  leading and trailing  ",
        "single",
        "a  b   c",
        "",
        "mix of   spaces here",
        "repeat   repeat",
        "x y",
    ]
    test_cases = [TestCase(args=[s], kwargs={}, expected=" ".join(reversed(s.split()))) for s in strings]
    code = f"""
def {fn}(text: str) -> str:
    return " ".join(reversed(text.split()))
"""
    return _task(
        task_id=task_id,
        difficulty=EASY,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["strings", "tokenization", "easy"],
        adversarial=True,
        template_name="reverse_words",
    )


def _easy_sum_even(rng: random.Random, task_id: str) -> Task:
    fn = f"sum_even_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(nums: list[int]) -> int:"
    instructions = "Return the sum of even integers in nums. Return 0 when no even numbers exist."

    def expected(values: list[int]) -> int:
        return sum(v for v in values if v % 2 == 0)

    candidates = [
        [],
        [1, 3, 5],
        [2, 4, 6],
        [-3, -2, -1, 0, 1],
        [10, 11, 12, 13, 14],
        [rng.randint(-30, 30) for _ in range(12)],
        [rng.randint(-5, 5) for _ in range(6)],
    ]
    test_cases = [TestCase(args=[arr], kwargs={}, expected=expected(arr)) for arr in candidates]
    code = f"""
def {fn}(nums: list[int]) -> int:
    return sum(v for v in nums if v % 2 == 0)
"""
    return _task(
        task_id=task_id,
        difficulty=EASY,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["lists", "filter", "easy"],
        adversarial=False,
        template_name="sum_even",
    )


def _medium_first_unique(_: random.Random, task_id: str) -> Task:
    fn = f"first_unique_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(text: str) -> int:"
    instructions = "Return the index of the first non-repeating character in text. Return -1 if none."

    def expected(s: str) -> int:
        counts: dict[str, int] = {}
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
        for idx, ch in enumerate(s):
            if counts[ch] == 1:
                return idx
        return -1

    cases = ["leetcode", "aabb", "swiss", "", "xxyz", "abba", "alphabet"]
    test_cases = [TestCase(args=[s], kwargs={}, expected=expected(s)) for s in cases]
    code = f"""
def {fn}(text: str) -> int:
    counts = {{}}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    for idx, ch in enumerate(text):
        if counts[ch] == 1:
            return idx
    return -1
"""
    return _task(
        task_id=task_id,
        difficulty=MEDIUM,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["strings", "hashmap", "medium"],
        adversarial=True,
        template_name="first_unique",
    )


def _medium_rotate_list(rng: random.Random, task_id: str) -> Task:
    fn = f"rotate_right_{task_id.lower().replace('-', '_')}"
    k = rng.randint(1, 14)
    signature = f"def {fn}(nums: list[int]) -> list[int]:"
    instructions = f"Rotate nums to the right by {k} positions and return a new list."

    def expected(values: list[int]) -> list[int]:
        if not values:
            return []
        shift = k % len(values)
        return values[-shift:] + values[:-shift]

    samples = [
        [],
        [1],
        [1, 2, 3],
        [4, 5, 6, 7],
        [9, 8, 7, 6, 5],
        [rng.randint(-10, 10) for _ in range(rng.randint(2, 8))],
    ]
    test_cases = [TestCase(args=[arr], kwargs={}, expected=expected(arr)) for arr in samples]
    code = f"""
def {fn}(nums: list[int]) -> list[int]:
    if not nums:
        return []
    shift = {k} % len(nums)
    return nums[-shift:] + nums[:-shift]
"""
    return _task(
        task_id=task_id,
        difficulty=MEDIUM,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["lists", "rotation", "medium"],
        adversarial=k > 7,
        template_name="rotate_list",
    )


def _medium_valid_brackets(_: random.Random, task_id: str) -> Task:
    fn = f"valid_brackets_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(text: str) -> bool:"
    instructions = "Return True if text contains a valid sequence of (), [], and {} brackets; otherwise False."

    def expected(text: str) -> bool:
        stack: list[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        for ch in text:
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                return False
        return not stack

    samples = ["", "()[]{}", "(]", "([{}])", "([)]", "{{{{", "{[()]}[]", ")("]
    test_cases = [TestCase(args=[s], kwargs={}, expected=expected(s)) for s in samples]
    code = f"""
def {fn}(text: str) -> bool:
    stack = []
    pairs = {{")": "(", "]": "[", "}}": "{{"}}
    for ch in text:
        if ch in "([{{":
            stack.append(ch)
        elif ch in ")]}}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            return False
    return not stack
"""
    return _task(
        task_id=task_id,
        difficulty=MEDIUM,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["stack", "parsing", "medium"],
        adversarial=True,
        template_name="valid_brackets",
    )


def _medium_lcp(rng: random.Random, task_id: str) -> Task:
    fn = f"longest_common_prefix_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(words: list[str]) -> str:"
    instructions = "Return the longest common prefix across all words. Return an empty string for empty input."

    def expected(words: list[str]) -> str:
        if not words:
            return ""
        prefix = words[0]
        for word in words[1:]:
            while not word.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    samples = [
        [],
        ["flower", "flow", "flight"],
        ["dog", "racecar", "car"],
        ["interview", "internet", "internal", "into"],
        ["same", "same"],
        ["", "blank"],
        [_rand_word(rng, 1, 4) for _ in range(4)],
    ]
    test_cases = [TestCase(args=[arr], kwargs={}, expected=expected(arr)) for arr in samples]
    code = f"""
def {fn}(words: list[str]) -> str:
    if not words:
        return ""
    prefix = words[0]
    for word in words[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
"""
    return _task(
        task_id=task_id,
        difficulty=MEDIUM,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["strings", "prefix", "medium"],
        adversarial=True,
        template_name="longest_common_prefix",
    )


def _medium_group_anagrams(rng: random.Random, task_id: str) -> Task:
    fn = f"group_anagrams_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(words: list[str]) -> list[list[str]]:"
    instructions = (
        "Group words that are anagrams. Each inner group must be sorted lexicographically. "
        "The final list of groups must be sorted by the first word in each group."
    )

    def expected(words: list[str]) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for word in words:
            key = "".join(sorted(word))
            groups.setdefault(key, []).append(word)
        out = [sorted(items) for items in groups.values()]
        out.sort(key=lambda grp: grp[0] if grp else "")
        return out

    base_words = [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat",
        "tab",
        _rand_word(rng, 3, 5),
    ]
    scrambled = base_words[:]
    rng.shuffle(scrambled)
    samples = [
        [],
        ["a"],
        ["ab", "ba", "abc"],
        scrambled,
        ["listen", "silent", "enlist", "inlets", "stone", "tones"],
    ]
    test_cases = [TestCase(args=[arr], kwargs={}, expected=expected(arr)) for arr in samples]
    code = f"""
def {fn}(words: list[str]) -> list[list[str]]:
    groups = {{}}
    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)
    out = [sorted(items) for items in groups.values()]
    out.sort(key=lambda grp: grp[0] if grp else "")
    return out
"""
    return _task(
        task_id=task_id,
        difficulty=MEDIUM,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["hashmap", "sorting", "medium"],
        adversarial=True,
        template_name="group_anagrams",
    )


def _hard_edit_distance(rng: random.Random, task_id: str) -> Task:
    fn = f"edit_distance_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(a: str, b: str) -> int:"
    instructions = "Return the Levenshtein edit distance between strings a and b."

    def expected(a: str, b: str) -> int:
        rows = len(a) + 1
        cols = len(b) + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[-1][-1]

    pairs = [
        ("", ""),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("abc", "abc"),
        (_rand_word(rng, 2, 5), _rand_word(rng, 2, 5)),
        (_rand_word(rng, 5, 8), _rand_word(rng, 4, 7)),
    ]
    test_cases = [TestCase(args=[a, b], kwargs={}, expected=expected(a, b)) for a, b in pairs]
    code = f"""
def {fn}(a: str, b: str) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]
"""
    return _task(
        task_id=task_id,
        difficulty=HARD,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["dp", "strings", "hard"],
        adversarial=True,
        template_name="edit_distance",
    )


def _hard_coin_change(rng: random.Random, task_id: str) -> Task:
    fn = f"coin_change_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(coins: list[int], amount: int) -> int:"
    instructions = (
        "Return the minimum number of coins needed to make amount. "
        "Return -1 if it is impossible."
    )

    def expected(coins: list[int], amount: int) -> int:
        limit = amount + 1
        dp = [limit] * (amount + 1)
        dp[0] = 0
        for value in range(1, amount + 1):
            for coin in coins:
                if coin <= value:
                    dp[value] = min(dp[value], dp[value - coin] + 1)
        return dp[amount] if dp[amount] != limit else -1

    samples = [
        ([1, 2, 5], 11),
        ([2], 3),
        ([1], 0),
        ([2, 4, 8], 15),
        ([3, 7, 10], 14),
        (sorted(set(rng.randint(1, 9) for _ in range(4))), rng.randint(6, 35)),
    ]
    test_cases = [
        TestCase(args=[coins, amount], kwargs={}, expected=expected(coins, amount))
        for coins, amount in samples
    ]
    code = f"""
def {fn}(coins: list[int], amount: int) -> int:
    limit = amount + 1
    dp = [limit] * (amount + 1)
    dp[0] = 0
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                dp[value] = min(dp[value], dp[value - coin] + 1)
    return dp[amount] if dp[amount] != limit else -1
"""
    return _task(
        task_id=task_id,
        difficulty=HARD,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["dp", "optimization", "hard"],
        adversarial=True,
        template_name="coin_change",
    )


def _hard_shortest_path(rng: random.Random, task_id: str) -> Task:
    fn = f"shortest_path_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(grid: list[list[int]]) -> int:"
    instructions = (
        "Given a grid of 0 (open) and 1 (blocked), return the shortest number of moves from "
        "top-left to bottom-right using 4-directional movement. Return -1 if unreachable."
    )

    def expected(grid: list[list[int]]) -> int:
        if not grid or not grid[0] or grid[0][0] == 1:
            return -1
        rows = len(grid)
        cols = len(grid[0])
        if grid[rows - 1][cols - 1] == 1:
            return -1
        from collections import deque

        queue = deque([(0, 0, 0)])
        seen = {(0, 0)}
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            r, c, dist = queue.popleft()
            if r == rows - 1 and c == cols - 1:
                return dist
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        return -1

    samples = [
        [[0]],
        [[1]],
        [[0, 0], [0, 0]],
        [[0, 1], [1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
    ]
    size = rng.randint(3, 5)
    random_grid = [[0 if rng.random() > 0.3 else 1 for _ in range(size)] for _ in range(size)]
    random_grid[0][0] = 0
    random_grid[-1][-1] = 0
    samples.append(random_grid)
    test_cases = [TestCase(args=[grid], kwargs={}, expected=expected(grid)) for grid in samples]
    code = f"""
def {fn}(grid: list[list[int]]) -> int:
    if not grid or not grid[0] or grid[0][0] == 1:
        return -1
    rows = len(grid)
    cols = len(grid[0])
    if grid[rows - 1][cols - 1] == 1:
        return -1
    from collections import deque

    queue = deque([(0, 0, 0)])
    seen = {{(0, 0)}}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        r, c, dist = queue.popleft()
        if r == rows - 1 and c == cols - 1:
            return dist
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    return -1
"""
    return _task(
        task_id=task_id,
        difficulty=HARD,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["graphs", "bfs", "hard"],
        adversarial=True,
        template_name="shortest_path_grid",
    )


def _hard_lis(rng: random.Random, task_id: str) -> Task:
    fn = f"lis_length_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(nums: list[int]) -> int:"
    instructions = "Return the length of the longest strictly increasing subsequence in nums."

    def expected(nums: list[int]) -> int:
        if not nums:
            return 0
        tails: list[int] = []
        from bisect import bisect_left

        for value in nums:
            idx = bisect_left(tails, value)
            if idx == len(tails):
                tails.append(value)
            else:
                tails[idx] = value
        return len(tails)

    samples = [
        [],
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 1, 0, 3, 2, 3],
        [7, 7, 7, 7],
        [4, 10, 4, 3, 8, 9],
        [rng.randint(-20, 30) for _ in range(rng.randint(6, 12))],
    ]
    test_cases = [TestCase(args=[arr], kwargs={}, expected=expected(arr)) for arr in samples]
    code = f"""
def {fn}(nums: list[int]) -> int:
    if not nums:
        return 0
    tails = []
    from bisect import bisect_left

    for value in nums:
        idx = bisect_left(tails, value)
        if idx == len(tails):
            tails.append(value)
        else:
            tails[idx] = value
    return len(tails)
"""
    return _task(
        task_id=task_id,
        difficulty=HARD,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["dp", "binary-search", "hard"],
        adversarial=True,
        template_name="lis_length",
    )


def _hard_word_ladder(rng: random.Random, task_id: str) -> Task:
    fn = f"word_ladder_{task_id.lower().replace('-', '_')}"
    signature = f"def {fn}(begin: str, end: str, words: list[str]) -> int:"
    instructions = (
        "Return the shortest transformation length from begin to end where each step changes one letter "
        "and must exist in words. Include both begin and end in length. Return 0 if impossible."
    )

    def expected(begin: str, end: str, words: list[str]) -> int:
        word_set = set(words)
        if end not in word_set:
            return 0
        from collections import deque

        queue = deque([(begin, 1)])
        seen = {begin}
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        while queue:
            word, depth = queue.popleft()
            if word == end:
                return depth
            for idx in range(len(word)):
                for letter in alphabet:
                    if letter == word[idx]:
                        continue
                    nxt = word[:idx] + letter + word[idx + 1 :]
                    if nxt in word_set and nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, depth + 1))
        return 0

    samples = [
        ("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]),
        ("hit", "cog", ["hot", "dot", "dog", "lot", "log"]),
        ("a", "c", ["a", "b", "c"]),
        ("same", "same", ["same"]),
    ]
    begin = "cold"
    end = "warm"
    words = ["cord", "card", "ward", "warm", "wold", "wald", "cold", "word"]
    if rng.random() > 0.5:
        words.append("worm")
    samples.append((begin, end, words))

    test_cases = [
        TestCase(args=[begin_word, end_word, vocab], kwargs={}, expected=expected(begin_word, end_word, vocab))
        for begin_word, end_word, vocab in samples
    ]
    code = f"""
def {fn}(begin: str, end: str, words: list[str]) -> int:
    word_set = set(words)
    if end not in word_set:
        return 0
    from collections import deque

    queue = deque([(begin, 1)])
    seen = {{begin}}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    while queue:
        word, depth = queue.popleft()
        if word == end:
            return depth
        for idx in range(len(word)):
            for letter in alphabet:
                if letter == word[idx]:
                    continue
                nxt = word[:idx] + letter + word[idx + 1:]
                if nxt in word_set and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
    return 0
"""
    return _task(
        task_id=task_id,
        difficulty=HARD,
        signature=signature,
        function_name=fn,
        instructions=instructions,
        canonical_solution=code,
        test_cases=test_cases,
        tags=["graphs", "strings", "hard"],
        adversarial=True,
        template_name="word_ladder",
    )


DIFFICULTY_TEMPLATES: dict[str, list[DifficultyTemplate]] = {
    EASY: [_easy_add_constant, _easy_count_char, _easy_clamp, _easy_reverse_words, _easy_sum_even],
    MEDIUM: [
        _medium_first_unique,
        _medium_rotate_list,
        _medium_valid_brackets,
        _medium_lcp,
        _medium_group_anagrams,
    ],
    HARD: [_hard_edit_distance, _hard_coin_change, _hard_shortest_path, _hard_lis, _hard_word_ladder],
}


@dataclass(frozen=True)
class DatasetSummary:
    total_tasks: int
    easy: int
    medium: int
    hard: int
    adversarial: int


def _bucket_counts(total_tasks: int) -> dict[str, int]:
    base = total_tasks // 3
    counts = {EASY: base, MEDIUM: base, HARD: base}
    remainder = total_tasks - (base * 3)
    order = [EASY, MEDIUM, HARD]
    for idx in range(remainder):
        counts[order[idx]] += 1
    return counts


def build_dataset(total_tasks: int = 600, seed: int = 7) -> list[Task]:
    if total_tasks < 1:
        raise ValueError("total_tasks must be > 0")
    counts = _bucket_counts(total_tasks)
    rng = random.Random(seed)
    tasks: list[Task] = []

    for difficulty in (EASY, MEDIUM, HARD):
        templates = DIFFICULTY_TEMPLATES[difficulty]
        for idx in range(counts[difficulty]):
            template = templates[idx % len(templates)]
            task_id = f"TCT-{difficulty[0].upper()}-{idx + 1:04d}"
            task_rng = random.Random(rng.randint(0, 10_000_000))
            tasks.append(template(task_rng, task_id))

    rng.shuffle(tasks)
    return tasks


def write_dataset(tasks: list[Task], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), separators=(",", ":")) + "\n")


def load_dataset(dataset_path: str | Path) -> list[Task]:
    path = Path(dataset_path)
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def dataset_summary(tasks: list[Task]) -> DatasetSummary:
    easy = sum(1 for task in tasks if task.difficulty == EASY)
    medium = sum(1 for task in tasks if task.difficulty == MEDIUM)
    hard = sum(1 for task in tasks if task.difficulty == HARD)
    adversarial = sum(1 for task in tasks if task.adversarial)
    return DatasetSummary(
        total_tasks=len(tasks),
        easy=easy,
        medium=medium,
        hard=hard,
        adversarial=adversarial,
    )


def _uniqueness_key(task: Task) -> str:
    payload = dict(task.to_dict())
    payload["task_id"] = "__TASK__"
    payload["function_name"] = "__FN__"
    payload["signature"] = str(payload.get("signature", "")).replace(task.function_name, "__FN__")
    payload["canonical_solution"] = str(payload.get("canonical_solution", "")).replace(
        task.function_name, "__FN__"
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_dataset_uniqueness(tasks: list[Task]) -> None:
    """Raise when duplicate logical tasks are found after ID/function-name normalization."""
    index_by_key: dict[str, int] = {}
    duplicates: list[tuple[str, str]] = []
    for idx, task in enumerate(tasks):
        key = _uniqueness_key(task)
        first_idx = index_by_key.get(key)
        if first_idx is None:
            index_by_key[key] = idx
            continue
        duplicates.append((tasks[first_idx].task_id, task.task_id))

    if duplicates:
        sample = ", ".join(f"{a}=={b}" for a, b in duplicates[:5])
        raise ValueError(
            f"Dataset contains {len(duplicates)} duplicate logical tasks. "
            f"Examples: {sample}"
        )


class TaskSampler:
    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def sample(
        self,
        n: int,
        *,
        difficulty: str | None = None,
        adversarial_only: bool = False,
        seed: int = 0,
    ) -> list[Task]:
        if n < 1:
            return []
        candidates = self._tasks
        if difficulty:
            candidates = [task for task in candidates if task.difficulty == difficulty]
        if adversarial_only:
            candidates = [task for task in candidates if task.adversarial]
        if not candidates:
            return []
        rng = random.Random(seed)
        if n >= len(candidates):
            out = candidates[:]
            rng.shuffle(out)
            return out
        return rng.sample(candidates, n)


def generate_and_write(
    output_path: str | Path,
    *,
    adversarial_output_path: str | Path | None = None,
    total_tasks: int = 600,
    seed: int = 7,
) -> DatasetSummary:
    tasks = build_dataset(total_tasks=total_tasks, seed=seed)
    validate_dataset_uniqueness(tasks)
    write_dataset(tasks, output_path)
    if adversarial_output_path is not None:
        write_dataset([task for task in tasks if task.adversarial], adversarial_output_path)
    return dataset_summary(tasks)
