\# How to set up your environment



Below is a \*\*step-by-step recipe\*\* you can follow exactly from your current folder:



`PS C:\\Users\\evertj\\git\\handson-ml3\\ch04>`



---



\## 0) One-time sanity check: do we have Python?



Run:



```powershell

py -V

```



If that prints a version (like `Python 3.12.x`), we’re golden.



If it errors, try:



```powershell

python --version

```



---



\## 1) Create a virtual environment in this folder



From:



`C:\\Users\\evertj\\git\\handson-ml3\\ch04`



Run:



```powershell

py -m venv .venv

```



That creates:



`ch04\\.venv\\`



---



\## 2) Activate the virtual environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



You should see your prompt change to include `(.venv)`.



If Windows whines about script execution policy, do this once in \*this PowerShell window\*:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

```



Then re-run the activate command.



---



\## 3) Upgrade pip inside the venv



```powershell

python -m pip install --upgrade pip

```



---



\## 4) Install exactly what we need for Jupyter + plotting



Start minimal and solid:



```powershell

python -m pip install notebook ipykernel matplotlib numpy pandas

```



Optional (nice visuals later):



```powershell

python -m pip install seaborn

```



---



\## 5) Register this venv as a Jupyter kernel (so VS Code can “see” it)



```powershell

python -m ipykernel install --user --name ch04-gd --display-name "Python (ch04 gradient descent)"

```



Now VS Code will have a kernel option called:

\*\*Python (ch04 gradient descent)\*\*



---



\## 6) Create the notebook file



In your `ch04` folder:



```powershell

ni Gradient\_Descent\_From\_Scratch.ipynb

```



(That just creates an empty notebook file placeholder.)



---



\## 7) Open VS Code in the current folder



```powershell

code .

```



---



\## 8) In VS Code: open notebook + pick kernel



1\. Click `Gradient\_Descent\_From\_Scratch.ipynb`

2\. VS Code will open it in notebook mode

3\. Top-right, click \*\*Select Kernel\*\*

4\. Choose: \*\*Python (ch04 gradient descent)\*\*



If you don’t see it:



\* Hit `Ctrl+Shift+P`

\* Search: \*\*Python: Select Interpreter\*\*

\* Pick the interpreter in: `...\\ch04\\.venv\\Scripts\\python.exe`

&nbsp; Then go back and select kernel again.



---



\# Phase 1.1 Notebook Cells (copy/paste)



\## ✅ Cell 1 (Markdown)



```markdown

\# Phase 1 — Build a Simple World



\## Step 1.1 — Create X values manually



\*\*Goal:\*\* Create a list of evenly spaced `x` values starting at 0.



We will:

1\. Start with `x = 0`

2\. Use a `for` loop to generate 10 values

3\. Store them in a list named `x\_values`



\*\*Concepts practiced:\*\*

\- Lists

\- Loops

\- Data representation

```



\## ✅ Cell 2 (Code)



```python

\# Step 1.1 — Create X values manually



x\_values = \[]          # empty list to fill

x = 0                  # start value

step = 1               # how much x increases each time



for \_ in range(10):    # repeat 10 times

&nbsp;   x\_values.append(x) # store current x

&nbsp;   x += step          # increase x



x\_values

```



\## ✅ Cell 3 (Markdown explanation)



```markdown

\### What just happened?



\- `x\_values = \[]` creates an empty list. We’ll fill it one value at a time.

\- `x = 0` sets our starting point.

\- `step = 1` means each new value will be 1 bigger than the last.



The loop:



\- `for \_ in range(10)` runs \*\*10 times\*\*

\- `x\_values.append(x)` pushes the current value of `x` into the list

\- `x += step` updates `x` for the next loop iteration



So we get:  

`\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

```



---



\## Quick “is everything alive?” test



In a new code cell in the notebook:



```python

import numpy as np

import matplotlib.pyplot as plt



print("Imports OK ✅")

```





