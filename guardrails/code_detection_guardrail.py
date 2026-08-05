def texts_to_scan(inputs, input_type):
    # For requests, scan only user-role messages so a code snippet in the
    # system prompt or a tool result never blocks the whole conversation.
    # SCAN_ALL_USER_TURNS=False checks only the newest user message, so a
    # previously-blocked code message lingering in client-side history
    # cannot re-block every later turn ("sticky block").
    SCAN_ALL_USER_TURNS = False

    msgs = inputs.get("structured_messages")
    if input_type == "request" and msgs:
        collected = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            if role != "user" and role != "USER" and role != "User":
                continue
            content = m.get("content")
            if isinstance(content, str):
                collected.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str):
                            collected.append(t)
        if len(collected) > 0:
            if SCAN_ALL_USER_TURNS:
                return collected
            return [collected[len(collected) - 1]]
    return inputs.get("texts", [])


def detect_code_rule(text):
    # flags: 10 = IGNORECASE|MULTILINE, 8 = MULTILINE only (case-sensitive)
    CI = 10
    CS = 8

    rules = [
        # fenced code blocks
        ("fence-lang", r"```[ \t]*(python|py|js|jsx|javascript|ts|tsx|typescript|java|c|cpp|hpp|cs|csharp|go|golang|rs|rust|rb|ruby|php|sql|sh|bash|zsh|shell|powershell|ps1|html|css|scss|json|yaml|yml|xml|kotlin|kt|swift|scala|perl|lua|dart|dockerfile|makefile|toml|diff)[ \t]*\r?\n", CI),
        ("fence-braces", r"```[\s\S]*?[{}][\s\S]*?```", CS),
        ("fence-semicolon-eol", r"```[\s\S]*?;\s*\n[\s\S]*?```", CS),
        # function definitions (paren glued to name; colon/brace/arrow required)
        ("py-def", r"\bdef\s+\w+\([^)]*\)\s*(->[^:]+)?:", CI),
        ("rb-def-linestart", r"^\s*def\s+\w+\(", CS),
        ("js-function-decl", r"\bfunction\s+\w+\s*\([^)]*\)\s*\{", CI),
        ("js-function-anon", r"\bfunction\s*\([^)]*\)\s*\{", CI),
        ("go-func", r"\bfunc\s+(\w+\s*\(|\()", CI),
        ("rust-fn", r"\bfn\s+\w+\([^)]*\)\s*(->|\{)", CS),
        ("kotlin-fun", r"\bfun\s+\w+\s*\([^)]*\)\s*\{", CS),
        # class / interface / enum
        ("class-brace", r"\bclass\s+[A-Z_]\w*\s*\{", CS),
        ("class-extends", r"\bclass\s+\w+\s+(extends|implements)\s+\w+", CS),
        ("py-class-base", r"\bclass\s+\w+\([\w\s,.]*\)\s*:", CI),
        ("interface-brace", r"\binterface\s+[A-Z]\w*\s*\{", CS),
        ("enum-brace", r"\benum\s+\w+\s*\{", CI),
        # java / c#
        ("java-class-def", r"\b(public|private|protected)\s+(static\s+|final\s+|abstract\s+)*(class|interface|enum)\s+\w+\s*(\{|extends\b|implements\b)", CI),
        ("java-method-sig", r"\b(public|private|protected)\s+(static\s+|final\s+)*(void|int|String|boolean|float|double|long|char)\s+\w+\s*\(", CI),
        ("java-psvm", r"\bpublic\s+static\s+void\s+main\b", CI),
        ("java-sysout", r"\bSystem\.(out|err)\.print", CS),
        ("cs-using-line", r"^using\s+[A-Z][\w.]*;\s*$", CS),
        ("cs-console", r"\bConsole\.Write(Line)?\s*\(", CS),
        # c / c++
        ("c-include", r"^\s*#include\s*[<\"]", CI),
        ("c-func-brace", r"\b(int|void|char|double|float|long|bool|unsigned)\s+\w+\s*\([^)]*\)\s*\{", CI),
        ("c-int-main", r"\bint\s+main\s*\(", CI),
        ("cpp-std", r"\bstd::\w+", CS),
        ("cpp-cout", r"\bcout\s*<<", CI),
        ("c-printf", r"\bprintf\s*\(", CI),
        ("cpp-using-ns", r"\busing\s+namespace\s+\w+", CI),
        # python
        ("py-print-quote", r"\bprint\s*\(\s*f?[\"']", CI),
        ("py-range-digit", r"\bfor\s+\w+\s+in\s+range\(\d", CI),
        ("py-import-line", r"^import\s+\w[\w.]*(\s+as\s+\w+)?\s*$", CS),
        ("py-from-import", r"^from\s+[\w.]+\s+import\s+\w+(\s*,\s*\w+)*(\s+as\s+\w+)?\s*$", CS),
        ("py-elif", r"^\s*elif\s+.+:", CS),
        ("py-except-typed", r"\bexcept\s+\w*(Error|Exception)\b", CS),
        ("py-return-const", r"\breturn\s+(None|True|False)\b", CS),
        ("py-dunder", r"\b__(init|name|main|file|dict|str|repr|len|call|iter|next|enter|exit|eq|hash|new|del|getattr|setattr|slots)__\b", CS),
        # javascript / typescript extras
        ("js-arrow-block", r"=>\s*\{", CI),
        ("js-arrow-empty", r"\(\)\s*=>", CI),
        ("js-console", r"\bconsole\.(log|error|warn|info)\s*\(", CI),
        ("js-require", r"\brequire\(['\"]", CI),
        ("js-document", r"\bdocument\.(getElementById|querySelector)", CS),
        # go / rust extras
        ("go-walrus-line", r"^\s*\w+(\s*,\s*\w+)*\s*:=\s*[&\[]?[\w.\"']+(\(\S*\))?(\s*[+\-*/%|&<>]+\s*\S+)*\s*$", CS),
        ("go-fmt", r"\bfmt\.Print(ln|f)?\s*\(", CS),
        ("rust-println", r"\bprintln!\s*\(", CI),
        ("rust-let-mut", r"\blet\s+mut\s+\w+", CS),
        # php
        ("php-open", r"<\?(php|=)", CI),
        ("php-arrow", r"\$[a-zA-Z_]\w*\s*->\s*[a-zA-Z_]", CS),
        ("php-echo-var", r"\becho\s+\$[a-zA-Z_]", CI),
        # sql (structure required; english-article guard below)
        ("sql-select-star", r"\bSELECT\s+\*\s+FROM\b", CI),
        ("sql-select-where-op", r"\bSELECT\b[\w\s,.*]+\bFROM\s+\w+\s+WHERE\s+.+(=|<|>|\bLIKE\b|\bIN\s*\(|\bIS\s+(NOT\s+)?NULL\b)", CI),
        ("sql-insert-values", r"\bINSERT\s+INTO\s+\w+\s*(\([\w\s,`\"]*\)\s*)?VALUES\s*\(", CI),
        ("sql-update-set", r"\bUPDATE\s+\w+\s+SET\s+\w+\s*=", CI),
        ("sql-delete-where-op", r"\bDELETE\s+FROM\s+\w+\s+WHERE\s+.+(=|<|>|\bLIKE\b|\bIN\s*\()", CI),
        ("sql-create-table", r"\bCREATE\s+TABLE\s+\w+\s*\([^)]*\b(INT|INTEGER|VARCHAR|TEXT|PRIMARY|SERIAL|CHAR|DATE|DECIMAL|BIGINT)\b", CI),
        ("sql-drop-table", r"\bDROP\s+TABLE\s+(IF\s+EXISTS\s+)?\w+\s*;", CI),
        ("sql-alter-table", r"\bALTER\s+TABLE\s+\w+\s+(ADD|DROP)\s+(COLUMN|CONSTRAINT|INDEX|PRIMARY|FOREIGN)\b", CI),
        ("sql-join", r"\b(INNER|OUTER|CROSS)\s+JOIN\b|\bJOIN\s+\w+\s+ON\s+\w+\.\w+\s*=", CI),
        # html / css
        ("html-tag", r"<(div|span|html|head|body|script|style|img|table|form|input|button|ul|ol|li|h[1-6])\b[^>]*>", CI),
        ("html-close", r"</(div|span|p|a|html|body|script|style|table|ul|ol|li|h[1-6])\s*>", CI),
        ("html-p-glued", r"<p\s*>|<p\s+class=", CI),
        ("html-a-href", r"<a\s+[^>]*href\s*=", CI),
        ("html-doctype", r"<!DOCTYPE\s+html", CI),
        ("css-prop-block", r"\{[^}]*\b(color|background|margin|padding|font|font-\w+|border|display|width|height|position|top|bottom|left|right|flex|grid|opacity|z-index|text-align|text-decoration|line-height|overflow)\s*:[^;}]+;[^}]*\}", CI),
        ("css-at-rule", r"@media\s*(\(|screen|only|print)|@keyframes\s+\w+|@font-face", CI),
        # shell (line-anchored or unambiguous syntax)
        ("sh-shebang", r"^#!\s*/", CS),
        ("sh-git-sub", r"\bgit\s+(clone|pull|push|commit|checkout|merge|rebase|status|stash|fetch|branch|log|diff|reset)\b", CI),
        ("sh-npm-line", r"^\s*(npm|pnpm|yarn)\s+(install|run|init|ci|build|test|start|publish)\b", CI),
        ("sh-pip-line", r"^\s*pip3?\s+(install|freeze|list|uninstall)\b", CI),
        ("sh-docker-line", r"^\s*(docker|docker-compose)\s+(run|build|pull|push|ps|exec|up|down|compose)\b", CI),
        ("sh-kubectl-line", r"^\s*kubectl\s+(get|apply|delete|describe|logs|exec)\b", CI),
        ("sh-pkg-line", r"^\s*(apt|apt-get|dnf|yum)\s+(install|update|upgrade|remove)\b", CI),
        ("sh-sudo-known", r"^\s*sudo\s+(apt|apt-get|yum|dnf|systemctl|service|docker|chmod|chown|rm|mkdir|cp|mv|npm|pip)\b", CI),
        ("sh-chmod-digits", r"\bchmod\s+[0-7][0-7][0-7]", CI),
        ("sh-rm-rf", r"\brm\s+-rf?\b", CI),
        ("sh-ls-flag", r"^\s*ls\s+-\w", CS),
        ("sh-echo-var", r"\becho\s+\$[a-zA-Z_{]", CI),
        ("sh-export", r"\bexport\s+\w+=\S", CI),
        ("sh-curl", r"\bcurl\s+(-\w|https?://)", CI),
        ("sh-pipe-cmd", r"\|\s*(grep|sed|awk|sort|uniq|head|tail|wc|xargs)\s", CI),
    ]

    # SELECT/DELETE ... WHERE with English articles between = prose, not SQL
    sql_guarded = ["sql-select-where-op", "sql-delete-where-op"]
    sql_prose = r"\b(SELECT|DELETE)\b[^\n]{0,120}\b(the|an|my|your|our|this|that|these|those)\b[^\n]{0,120}\bWHERE\b"

    for rule in rules:
        name = rule[0]
        pattern = rule[1]
        flags = rule[2]
        if regex_match(text, pattern, flags):
            is_guarded = False
            for g in sql_guarded:
                if g == name:
                    is_guarded = True
            if is_guarded and regex_match(text, sql_prose, CI):
                continue
            return name

    # structural: >=2 lines ending in ');' or '};' (C/Java/JS statement lines)
    if len(regex_find_all(text, r"[)}]\s*;\s*$", CS)) >= 2:
        return "struct-semicolon-lines"
    # structural: '{' ends a line AND a later line starts with '}'
    if regex_match(text, r"\{\s*$", CS) and regex_match(text, r"^\s*\}", CS):
        return "struct-brace-pair"
    return None


def apply_guardrail(inputs, request_data, input_type):
    texts = texts_to_scan(inputs, input_type)
    idx = 0
    for text in texts:
        if isinstance(text, str):
            rule = detect_code_rule(text)
            if rule is not None:
                return block(
                    "Sharing or requesting code is not permitted on this service.",
                    {"rule": rule, "direction": input_type, "text_index": idx},
                )
        idx = idx + 1
    return allow()
