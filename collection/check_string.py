import tkinter as tk
from tkinter import ttk

def validate():
    s = entry.get()

    valid_chars = len(s) > 0 and all(c in "ab" for c in s)
    odd_length = len(s) % 2 == 1

    rule1.set("✔ Valid symbols (a, b)" if valid_chars else "✘ Invalid symbols (a, b)")
    rule2.set("✔ Odd length" if odd_length else "✘ Even length")
    
    rule1_label.config(fg="#2ecc71" if valid_chars else "#e74c3c")
    rule2_label.config(fg="#2ecc71" if odd_length else "#e74c3c")

    if valid_chars and odd_length:
        result.set("✓ ACCEPTED")
        result_label.config(fg="#27ae60", bg="#d5f4e6", padx=10)
    else:
        result.set("✗ REJECTED")
        result_label.config(fg="#c0392b", bg="#fadbd8", padx=10)

root = tk.Tk()
root.title("String Validator - Odd Length {a,b}*")
root.geometry("450x400")
root.configure(bg="#f5f6fa")
root.resizable(False, False)

# Header
header_frame = tk.Frame(root, bg="#3498db", height=60)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

tk.Label(header_frame, text="String Validator", 
         font=("Segoe UI", 16, "bold"), bg="#3498db", fg="white").pack(pady=15)

# Main content frame
content_frame = tk.Frame(root, bg="#f5f6fa")
content_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Input section
input_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
input_frame.pack(fill="x", pady=(0, 15))

tk.Label(input_frame, text="Enter String:", font=("Segoe UI", 10), 
         bg="white", fg="#2c3e50").pack(anchor="w", padx=15, pady=(10, 5))

entry = tk.Entry(input_frame, font=("Segoe UI", 12), relief="flat", 
                 bg="#ecf0f1", fg="#2c3e50")
entry.pack(fill="x", padx=15, pady=(0, 10))
entry.focus()

# Validate button
validate_btn = tk.Button(content_frame, text="Validate String", command=validate,
                         font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white",
                         relief="flat", cursor="hand2", padx=20, pady=8)
validate_btn.pack(pady=(0, 15))
validate_btn.bind("<Enter>", lambda e: validate_btn.config(bg="#2980b9"))
validate_btn.bind("<Leave>", lambda e: validate_btn.config(bg="#3498db"))

# Rules section
rules_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
rules_frame.pack(fill="x", pady=(0, 15))

tk.Label(rules_frame, text="Validation Rules:", font=("Segoe UI", 10, "bold"), 
         bg="white", fg="#2c3e50").pack(anchor="w", padx=15, pady=(10, 5))

rule1 = tk.StringVar(value="✘ Valid symbols (a, b)")
rule2 = tk.StringVar(value="✘ Odd length")

rule1_label = tk.Label(rules_frame, textvariable=rule1, anchor="w", 
                       font=("Segoe UI", 10), bg="white", fg="#e74c3c")
rule1_label.pack(fill="x", padx=15, pady=3)

rule2_label = tk.Label(rules_frame, textvariable=rule2, anchor="w", 
                       font=("Segoe UI", 10), bg="white", fg="#e74c3c")
rule2_label.pack(fill="x", padx=15, pady=(3, 10))

# Result section
result = tk.StringVar()
result_label = tk.Label(content_frame, textvariable=result, 
                       font=("Segoe UI", 14, "bold"), bg="#f5f6fa",
                       relief="solid", bd=1, pady=10)
result_label.pack(fill="x")

# Info footer
tk.Label(content_frame, text="L = { w | w ∈ {a,b}* and |w| is odd }", 
         font=("Segoe UI", 8), bg="#f5f6fa", fg="#7f8c8d").pack(pady=(10, 0))

root.mainloop()
