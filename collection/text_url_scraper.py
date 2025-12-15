import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from urllib.parse import urlparse
import re

URL_CANDIDATE_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\)]+"
)


def _normalize_for_validation(url: str) -> str:
    url = url.strip()
    if url.lower().startswith("www."):
        return "https://" + url
    return url


def _is_valid_url(url: str) -> bool:
    normalized = _normalize_for_validation(url)
    try:
        parsed = urlparse(normalized)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if "." not in parsed.netloc:
        return False
    return True


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for match in URL_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).rstrip(".,;:!?)\"]}'")
        if candidate and candidate not in seen:
            seen.add(candidate)
            results.append(candidate)
    return results


class UrlScraperApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("URL Scraper - Extract URLs from Text")
        root.geometry("650x600")
        root.configure(bg="#f5f6fa")

        header_frame = tk.Frame(root, bg="#8e44ad", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="URL Scraper",
            font=("Segoe UI", 16, "bold"),
            bg="#8e44ad",
            fg="white",
        ).pack(pady=15)

        content_frame = tk.Frame(root, bg="#f5f6fa")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        input_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
        input_frame.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(
            input_frame,
            text="Enter Text:",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#2c3e50",
        ).pack(anchor="w", padx=15, pady=(10, 5))

        input_scroll = tk.Scrollbar(input_frame)
        input_scroll.pack(side="right", fill="y", padx=(0, 5), pady=(0, 10))

        self.input_box = tk.Text(
            input_frame,
            height=6,
            font=("Segoe UI", 10),
            relief="flat",
            bg="#ecf0f1",
            fg="#2c3e50",
            yscrollcommand=input_scroll.set,
            wrap="word",
        )
        self.input_box.pack(fill="both", expand=True, padx=(15, 0), pady=(0, 10))
        self.input_box.bind("<KeyRelease>", self.on_text_change)
        input_scroll.config(command=self.input_box.yview)

        valid_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
        valid_frame.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(
            valid_frame,
            text="Valid URLs",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#27ae60",
        ).pack(anchor="w", padx=15, pady=(10, 5))

        valid_scroll = tk.Scrollbar(valid_frame)
        valid_scroll.pack(side="right", fill="y", padx=(0, 5), pady=(0, 10))

        self.valid_list = tk.Listbox(
            valid_frame,
            height=5,
            font=("Segoe UI", 9),
            relief="flat",
            bg="#e8f8f5",
            fg="#27ae60",
            yscrollcommand=valid_scroll.set,
            selectbackground="#27ae60",
        )
        self.valid_list.pack(fill="both", expand=True, padx=(15, 0), pady=(0, 10))
        valid_scroll.config(command=self.valid_list.yview)

        invalid_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
        invalid_frame.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(
            invalid_frame,
            text="Invalid URLs",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#e74c3c",
        ).pack(anchor="w", padx=15, pady=(10, 5))

        invalid_scroll = tk.Scrollbar(invalid_frame)
        invalid_scroll.pack(side="right", fill="y", padx=(0, 5), pady=(0, 10))

        self.invalid_list = tk.Listbox(
            invalid_frame,
            height=3,
            font=("Segoe UI", 9),
            relief="flat",
            bg="#fadbd8",
            fg="#c0392b",
            yscrollcommand=invalid_scroll.set,
            selectbackground="#e74c3c",
        )
        self.invalid_list.pack(fill="both", expand=True, padx=(15, 0), pady=(0, 10))
        invalid_scroll.config(command=self.invalid_list.yview)

        stats_frame = tk.Frame(content_frame, bg="#f5f6fa")
        stats_frame.pack(fill="x")

        self.count = tk.StringVar(value="Total: 0 URLs  |  Valid: 0  |  Invalid: 0")
        tk.Label(
            stats_frame,
            textvariable=self.count,
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            fg="#34495e",
        ).pack(pady=(0, 10))

        buttons_frame = tk.Frame(stats_frame, bg="#f5f6fa")
        buttons_frame.pack()

        self.save_btn = tk.Button(
            buttons_frame,
            text="Save URLs",
            command=self.save_urls,
            font=("Segoe UI", 10, "bold"),
            bg="#27ae60",
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=8,
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=5)
        self.save_btn.bind(
            "<Enter>",
            lambda e: self.save_btn.config(bg="#229954")
            if self.save_btn["state"] == "normal"
            else None,
        )
        self.save_btn.bind(
            "<Leave>",
            lambda e: self.save_btn.config(bg="#27ae60")
            if self.save_btn["state"] == "normal"
            else None,
        )

        clear_btn = tk.Button(
            buttons_frame,
            text="Clear All",
            command=self.clear_all,
            font=("Segoe UI", 10, "bold"),
            bg="#95a5a6",
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=8,
        )
        clear_btn.pack(side="left", padx=5)
        clear_btn.bind("<Enter>", lambda e: clear_btn.config(bg="#7f8c8d"))
        clear_btn.bind("<Leave>", lambda e: clear_btn.config(bg="#95a5a6"))

        self.valid_urls: list[str] = []
        self.invalid_urls: list[str] = []

    def on_text_change(self, _event: object | None = None) -> None:
        text = self.input_box.get("1.0", tk.END)
        candidates = _extract_urls(text)

        valid: list[str] = []
        invalid: list[str] = []
        for u in candidates:
            (valid if _is_valid_url(u) else invalid).append(u)

        self.valid_urls = valid
        self.invalid_urls = invalid

        self.valid_list.delete(0, tk.END)
        self.invalid_list.delete(0, tk.END)

        for u in valid:
            self.valid_list.insert(tk.END, u)
        for u in invalid:
            self.invalid_list.insert(tk.END, u)

        self.count.set(
            f"Total: {len(candidates)} URLs  |  Valid: {len(valid)}  |  Invalid: {len(invalid)}"
        )

        self.save_btn.config(state="normal" if valid else "disabled")

    def save_urls(self) -> None:
        if not self.valid_urls:
            messagebox.showwarning("No URLs", "No valid URLs to save.")
            return

        filename = f"urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, "w", encoding="utf-8", newline="\n") as f:
                for u in self.valid_urls:
                    f.write(u + "\n")
            messagebox.showinfo("Saved", f"Saved {len(self.valid_urls)} URLs to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save URLs:\n{e}")

    def clear_all(self) -> None:
        self.input_box.delete("1.0", tk.END)
        self.valid_list.delete(0, tk.END)
        self.invalid_list.delete(0, tk.END)
        self.valid_urls = []
        self.invalid_urls = []
        self.count.set("Total: 0 URLs  |  Valid: 0  |  Invalid: 0")
        self.save_btn.config(state="disabled")


def main() -> None:
    root = tk.Tk()
    UrlScraperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
