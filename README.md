LLM council with 3 solvers and 1 judge that follow these steps:
0. A problem is given to the 3 solvers, independently of each other
1. Each proposes their solution
2. Each solver gives feedback on other 2's solutions
3. Each solver considers feedback and adjusts its own solution
4. Judge decides which final solution it likes most and gives it as response

## Council Checking Script
Example CLI run

`python scripts/council_preview.py "Design a robust cache invalidation plan for a CDN-backed API."`

[Result](council_checking_script_run_result.md)
