"""
Test suite for guardrails/code_detection_guardrail.py

Runs the guardrail source in an environment mirroring LiteLLM's custom-code
sandbox primitives (regex_match / regex_find_all / allow / block), then
validates it against the full corpus:

  - MUST_BLOCK: real code across ~15 languages + shell + SQL + fenced blocks
  - MUST_ALLOW: conversational / business messages, including every
    false-positive trap found while hardening (overlapping English words
    like class/function/select/import, casual no-space parens, emoticons,
    braces-in-prose, etc.)
  - Role-scoping: system-prompt code must not block; latest user-turn code
    must block; stale blocked code in history must not re-block.

Usage:
    python3 test_code_detection_guardrail.py

If RestrictedPython is installed (pip install RestrictedPython), the source
is additionally compiled with LiteLLM's actual sandbox compiler to catch
constructs the proxy would reject (e.g. leading-underscore names).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GUARDRAIL_PATH = os.path.join(HERE, "code_detection_guardrail.py")


# ---- primitives, mirroring litellm custom_code/primitives.py ----
def allow():
    return {"action": "allow"}


def block(reason, detection_info=None):
    result = {"action": "block", "reason": reason}
    if detection_info:
        result["detection_info"] = detection_info
    return result


def regex_match(text, pattern, flags=0):
    try:
        return bool(re.search(pattern, text, flags))
    except re.error:
        return False


def regex_find_all(text, pattern, flags=0):
    try:
        return re.findall(pattern, text, flags)
    except re.error:
        return []


def load_guardrail():
    with open(GUARDRAIL_PATH) as f:
        source = f.read()

    env = {
        "allow": allow,
        "block": block,
        "regex_match": regex_match,
        "regex_find_all": regex_find_all,
        "len": len, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "isinstance": isinstance,
    }

    # Optional: compile with LiteLLM's actual sandbox compiler if available
    try:
        from RestrictedPython import (
            RestrictingNodeTransformer,
            compile_restricted,
            limited_builtins,
            safe_builtins,
            utility_builtins,
        )
        from RestrictedPython.Eval import (
            default_guarded_getitem,
            default_guarded_getiter,
        )
        from RestrictedPython.Guards import (
            full_write_guard,
            guarded_iter_unpack_sequence,
            safer_getattr,
        )
        import operator

        class AsyncAwareTransformer(RestrictingNodeTransformer):
            def visit_AsyncFunctionDef(self, node):
                return self.visit_FunctionDef(node)

            def visit_AsyncFor(self, node):
                return self.node_contents_visit(node)

            def visit_AsyncWith(self, node):
                return self.node_contents_visit(node)

            def visit_Await(self, node):
                return self.node_contents_visit(node)

        ops = {"+=": operator.iadd, "-=": operator.isub, "*=": operator.imul}
        env["__builtins__"] = {**safe_builtins, **limited_builtins, **utility_builtins}
        env["_getattr_"] = safer_getattr
        env["_getitem_"] = default_guarded_getitem
        env["_getiter_"] = default_guarded_getiter
        env["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        env["_write_"] = full_write_guard
        env["_inplacevar_"] = lambda op, x, y: ops[op](x, y)
        compiled = compile_restricted(
            source=source, filename="<guardrail>", mode="exec",
            policy=AsyncAwareTransformer,
        )
        print("Compiled with RestrictedPython sandbox (same as LiteLLM proxy)")
    except ImportError:
        compiled = compile(source, "<guardrail>", "exec")
        print("RestrictedPython not installed - compiled with plain Python")
        print("(pip install RestrictedPython for full sandbox-compat check)")

    exec(compiled, env)
    return env["apply_guardrail"]


MUST_BLOCK = [
    # python
    "def get_user(id):\n    return db.query(id)",
    "class Foo(Base):\n    def __init__(self):\n        pass",
    "import numpy as np",
    "from os import path",
    "for i in range(10): print(i)",
    'print("hello world")',
    'if __name__ == "__main__":\n    main()',
    "try:\n    x = 1\nexcept ValueError:\n    pass",
    # javascript / typescript
    "function add(a, b) { return a + b; }",
    "const f = (x) => { return x * 2; };",
    'console.log("hi");',
    "interface User { name: string; age: number; }",
    "document.getElementById('app')",
    "const express = require('express');",
    # java / c#
    "public class Main { public static void main(String[] args) {} }",
    'System.out.println("hello");',
    "using System;\nConsole.WriteLine(\"hi\");",
    # c / c++
    '#include <stdio.h>\nint main(void) {\n    printf("hi");\n    return 0;\n}',
    'std::cout << "hello" << std::endl;',
    "using namespace std;",
    # go
    'func main() {\n\tfmt.Println("hi")\n}',
    "x := compute()\ny := x * 2",
    # rust
    'fn main() {\n    println!("hi");\n}',
    "let mut count = 0;",
    # sql
    "SELECT * FROM users WHERE id = 1",
    "SELECT name, email FROM users WHERE status = 'active'",
    "INSERT INTO users (name) VALUES ('alice')",
    "UPDATE users SET name = 'bob' WHERE id = 2",
    "DELETE FROM sessions WHERE expires < now()",
    "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50))",
    "DROP TABLE old_logs;",
    "SELECT a.id FROM orders a INNER JOIN users b ON a.uid = b.id",
    # html / css
    '<div class="container"><p>hi</p></div>',
    '<a href="https://example.com">link</a>',
    ".container { color: red; margin: 0; }",
    "@media (max-width: 600px) { body { font-size: 12px; } }",
    # shell
    '#!/bin/bash\necho "hello"',
    "git clone https://github.com/foo/bar.git",
    "npm install express",
    "pip install requests",
    "chmod 755 deploy.sh",
    "rm -rf ./build",
    "export PATH=/usr/local/bin:$PATH",
    "echo $HOME",
    "curl -X POST https://api.example.com/v1",
    "cat access.log | grep 500 | wc -l",
    "sudo apt install nginx",
    # php / kotlin
    '<?php echo "hello"; ?>',
    '$user->name = "bob";',
    'fun main() {\n    println("hi")\n}',
    # fenced
    '```python\nprint("hi")\n```',
    "```\nlet total = 0;\nfor (const x of items) {\n  total += x;\n}\n```",
    # C-family multiline
    "int square(int x) {\n    return x * x;\n}",
]

MUST_ALLOW = [
    "Please select an item from the list below.",
    "My Python class starts at 5pm on Monday.",
    "Can you explain what a function is in math?",
    "{ note: this is important; please read }",
    "Run npm install and restart the server.",
    "What's the weather like today?",
    "",
    "Please set (as discussed) the deadline for Friday.",
    "See the list (attached) for details.",
    "The value falls within a range (approximately 10-20).",
    "Please print (double-sided) the report.",
    "Check the class times: Monday and Wednesday evenings.",
    "Select your plan from below: Basic = $10, Plus = $20.",
    "Please pick a class schedule that works for you.",
    "call me later(around 6pm)",
    "check the list(attached) when you get a chance",
    "I'll set(a reminder) for you",
    "just print(the tickets) when ready pls",
    "here's the full list(everything included)",
    "Select any topic you'd like from the menu; I'm happy to help.",
    "{fun fact: I love pizza; ask me anything}",
    "Hey! How was your weekend?",
    "I'm feeling really overwhelmed with work lately.",
    "Can you recommend a good book to read on vacation?",
    "What's your favorite type of pizza?",
    "I think we should meet at 3pm instead of 2pm.",
    "lol that's so funny, tell me more",
    "omg I can't believe that happened",
    "Thanks for your help earlier, really appreciate it!",
    "I want to learn Python programming, where do I start?",
    "Can you explain how a for loop works conceptually?",
    "What's the difference between a list and a set in general terms?",
    "Is JavaScript hard to learn for beginners?",
    "My computer class starts next week.",
    "The printer is out of paper again.",
    "Can you print this document for me?",
    "I set a goal to read more this year.",
    "Let's set up a meeting for next week.",
    "Please set the table before dinner.",
    "I need to reset my password.",
    "What's on your reading list this month?",
    "The range of the new phone's battery is impressive.",
    "That functions well for our needs.",
    "This class of problems is really interesting.",
    "What's 5 * 3 + 2?",
    "The meeting is scheduled for 3:30; please be on time.",
    "Grades range from A to F; most students got a B.",
    "Ugh, Mondays :=( am I right?",
    "Mondays :=D am I right",
    "See you soon! <3",
    "Ratio: 3:2, easy to remember.",
    "she teaches a public class on ethics every semester",
    "I booked economy class seats (row 12) for us",
    "I offer private class sessions for yoga",
    "the function hall (booked) for the wedding",
    "we hired the function room(s) near the station",
    "I def want (really) to come tonight",
    "Please select a dish from the menu where vegetarian options are marked",
    "select the winner from the list where scores are highest",
    "insert into the schedule a 15-minute break",
    "please delete from the list whatever is expired",
    "the echo 'sounded' amazing in that hall",
    "meeting notes {agenda: budget; time: 3pm}",
    "review the design {header: navy; footer: gray}",
    "Import duties as well as taxes apply.",
    "@john are you coming tonight?",
    "that was __awesome__ tonight",
    "money => happiness, simple as that",
    "(P and Q) => R in propositional logic",
    "Make sure to check the docs first",
    "Brew some coffee and let's chat",
    "Curl up with a good book this weekend",
    "Make dinner plans (7pm?)",
    "my cd / dvd collection is for sale",
    "our get-together is at my place",
    "Try: restarting the app first",
    "    if you have any questions, reach out anytime",
    "From Spain, import the best olives every season",
    "the score went 5->3 in the second half",
    "monday->friday commute is brutal",
    "my answer = True on the quiz, was I right?",
    "if it == free, I'm in lol",
    "The package main issue is the delivery time",
    "Console gaming > PC gaming, fight me",
    "It costs $50 = about €45",
    "Milk;\nEggs;\nBread;",
    "the class {2024} reunion is coming up",
    "use the fn keys (F1-F12) to adjust brightness",
    "had so much fun playing (games) with you all",
    "have fun coding (later tonight)",
    "fun playing (games) = best day ever",
    "we should drop table tennis from the club schedule",
    "can you create table reservations (for 8pm)?",
    "alter table arrangements, add more chairs please",
    "Dear Sir;\nThanks for your patience;",
    "f(x) = 2x + 1 is a linear function",
    "y = mx + b, remember?",
    "H2O + NaCl -> salt water",
    "price went $50->60 overnight",
    "5 < p > 2 doesn't make sense as an inequality",
    "sudo make me a sandwich",
    "then he was like ¯\\_(ツ)_/¯",
    "update the roster for Monday's game",
    "you can update your address in settings",
    "the update set everyone off",
    "Schedule {Monday: gym; Tuesday: yoga}",
    "left join us for dinner if you're free",
    "grep through your feelings later, we have work",
    "Git is a version control system, btw.",
    "Def Leppard tickets go on sale Friday!",
    "echo chamber effects are real; be careful out there",
]


def main():
    apply_guardrail = load_guardrail()

    fp, fn = [], []
    for t in MUST_BLOCK:
        r = apply_guardrail({"texts": [t]}, {}, "request")
        if r["action"] != "block":
            fn.append(t)
    for t in MUST_ALLOW:
        r = apply_guardrail({"texts": [t]}, {}, "request")
        if r["action"] != "allow":
            fp.append((t, r.get("detection_info")))

    # role-scoping checks
    checks = []
    r = apply_guardrail({
        "texts": ["```json\n{\"a\": 1}\n```", "What's 2+2?"],
        "structured_messages": [
            {"role": "system", "content": "Reply as: ```json\n{\"a\": 1}\n```"},
            {"role": "user", "content": "What's 2+2?"},
        ]}, {}, "request")
    checks.append(("system-prompt code ignored", r["action"] == "allow"))

    r = apply_guardrail({
        "texts": ["hi", "def foo(x):\n    return x"],
        "structured_messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "def foo(x):\n    return x"},
        ]}, {}, "request")
    checks.append(("latest user code blocked", r["action"] == "block"))

    r = apply_guardrail({
        "texts": ["def foo(x):\n    return x", "what's the weather?"],
        "structured_messages": [
            {"role": "user", "content": "def foo(x):\n    return x"},
            {"role": "assistant", "content": "Not permitted."},
            {"role": "user", "content": "ok never mind, what's the weather?"},
        ]}, {}, "request")
    checks.append(("stale blocked code not sticky", r["action"] == "allow"))

    r = apply_guardrail(
        {"texts": ["Sure:\n```python\nprint('hi')\n```"]}, {}, "response")
    checks.append(("response-side code blocked", r["action"] == "block"))

    print()
    print("MUST_BLOCK: %d/%d blocked" % (len(MUST_BLOCK) - len(fn), len(MUST_BLOCK)))
    print("MUST_ALLOW: %d/%d allowed" % (len(MUST_ALLOW) - len(fp), len(MUST_ALLOW)))
    for name, ok in checks:
        print("%s: %s" % ("PASS" if ok else "FAIL", name))
    for t, info in fp:
        print("  FALSE POSITIVE: %r -> %s" % (t, info))
    for t in fn:
        print("  MISSED: %r" % t)

    failed = fp or fn or not all(ok for _, ok in checks)
    print("\n%s" % ("FAILED" if failed else "ALL PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
