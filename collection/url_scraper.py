import tkinter as tk
from tkinter import messagebox, scrolledtext
import re
import csv
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

scraped_data = None

def scrape_website():
    url = url_entry.get().strip()
    
    if not DEPENDENCIES_INSTALLED:
        messagebox.showerror("Missing Dependencies", 
                           "Please install required packages:\npip install requests beautifulsoup4")
        return
    
    if not url:
        messagebox.showwarning("Empty URL", "Please enter a URL to scrape!")
        return
    
    # Add http:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Clear previous results
    output_text.delete("1.0", tk.END)
    output_text.insert("1.0", "Scraping... Please wait...\n")
    scrape_btn.config(state="disabled")
    root.update()
    
    try:
        # Fetch the webpage
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        title = soup.title.string.strip() if soup.title else "No title found"
        
        # Get all links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                links.append(href)
        links = list(set(links))[:20]  # Limit to 20 unique links
        
        # Get all emails
        emails = set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response.text))
        
        # Get headings
        headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3']) if h.get_text().strip()]
        headings = headings[:10]  # Limit to 10 headings
        
        # Display results
        output_text.delete("1.0", tk.END)
        output_text.insert("1.0", f"Successfully scraped: {url}\n")
        output_text.insert(tk.END, "="*70 + "\n\n")
        
        output_text.insert(tk.END, f"TITLE:\n{title}\n\n")
        
        output_text.insert(tk.END, f"HEADINGS ({len(headings)} found):\n")
        for i, heading in enumerate(headings, 1):
            output_text.insert(tk.END, f"  {i}. {heading}\n")
        output_text.insert(tk.END, "\n")
        
        output_text.insert(tk.END, f"EMAILS ({len(emails)} found):\n")
        if emails:
            for email in list(emails)[:10]:
                output_text.insert(tk.END, f"  - {email}\n")
        else:
            output_text.insert(tk.END, "  (No emails found)\n")
        output_text.insert(tk.END, "\n")
        
        output_text.insert(tk.END, f"LINKS ({len(links)} found):\n")
        for i, link in enumerate(links, 1):
            output_text.insert(tk.END, f"  {i}. {link}\n")
        
        # Store data for saving
        global scraped_data
        scraped_data = {
            'url': url,
            'title': title,
            'headings': headings,
            'emails': list(emails),
            'links': links,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_btn.config(state="normal")
        status_label.config(text=f"Scraped successfully | Found: {len(links)} links, {len(emails)} emails", 
                          fg="#27ae60")
        
    except requests.exceptions.RequestException as e:
        output_text.delete("1.0", tk.END)
        output_text.insert("1.0", f"Error scraping website:\n\n{str(e)}\n\n")
        output_text.insert(tk.END, "Tips:\n")
        output_text.insert(tk.END, "- Check if the URL is correct\n")
        output_text.insert(tk.END, "- Make sure you have internet connection\n")
        output_text.insert(tk.END, "- Some websites block scrapers\n")
        status_label.config(text="Scraping failed", fg="#e74c3c")
    except Exception as e:
        output_text.delete("1.0", tk.END)
        output_text.insert("1.0", f"Unexpected error:\n\n{str(e)}")
        status_label.config(text="Error occurred", fg="#e74c3c")
    finally:
        scrape_btn.config(state="normal")

def save_data():
    if not scraped_data:
        messagebox.showwarning("No Data", "No data to save! Scrape a website first.")
        return
    
    try:
        # Save to CSV
        filename = f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Type', 'Data'])
            writer.writerow(['URL', scraped_data['url']])
            writer.writerow(['Title', scraped_data['title']])
            writer.writerow(['Timestamp', scraped_data['timestamp']])
            writer.writerow([])
            
            writer.writerow(['Headings', ''])
            for heading in scraped_data['headings']:
                writer.writerow(['', heading])
            writer.writerow([])
            
            writer.writerow(['Emails', ''])
            for email in scraped_data['emails']:
                writer.writerow(['', email])
            writer.writerow([])
            
            writer.writerow(['Links', ''])
            for link in scraped_data['links']:
                writer.writerow(['', link])
        
        messagebox.showinfo("Success", f"Data saved to:\n{filename}")
        status_label.config(text=f"✓ Data saved to {filename}", fg="#27ae60")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data:\n{str(e)}")

def clear_all():
    url_entry.delete(0, tk.END)
    output_text.delete("1.0", tk.END)
    save_btn.config(state="disabled")
    status_label.config(text="Ready to scrape", fg="#7f8c8d")
    global scraped_data
    scraped_data = None

root = tk.Tk()
root.title("Web Scraper - Extract Data from Websites")
root.geometry("700x650")
root.configure(bg="#f5f6fa")

# Header
header_frame = tk.Frame(root, bg="#2c3e50", height=70)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

tk.Label(header_frame, text="Web Scraper", 
         font=("Segoe UI", 18, "bold"), bg="#2c3e50", fg="white").pack(pady=10)
# Main content
content_frame = tk.Frame(root, bg="#f5f6fa")
content_frame.pack(fill="both", expand=True, padx=20, pady=20)

# URL Input section
input_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
input_frame.pack(fill="x", pady=(0, 15))

tk.Label(input_frame, text="Website URL:", font=("Segoe UI", 10, "bold"), 
         bg="white", fg="#2c3e50").pack(anchor="w", padx=15, pady=(12, 5))

url_input_frame = tk.Frame(input_frame, bg="white")
url_input_frame.pack(fill="x", padx=15, pady=(0, 12))

url_entry = tk.Entry(url_input_frame, font=("Segoe UI", 11), relief="flat", 
                     bg="#ecf0f1", fg="#2c3e50")
url_entry.pack(side="left", fill="x", expand=True, ipady=5)
url_entry.insert(0, "example.com")
url_entry.bind('<Return>', lambda e: scrape_website())

scrape_btn = tk.Button(url_input_frame, text="Scrape", command=scrape_website,
                       font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white",
                       relief="flat", cursor="hand2", padx=20)
scrape_btn.pack(side="left", padx=(10, 0))
scrape_btn.bind("<Enter>", lambda e: scrape_btn.config(bg="#2980b9"))
scrape_btn.bind("<Leave>", lambda e: scrape_btn.config(bg="#3498db"))

# Output section
output_frame = tk.Frame(content_frame, bg="white", relief="solid", bd=1)
output_frame.pack(fill="both", expand=True, pady=(0, 15))

tk.Label(output_frame, text="Scraped Data:", font=("Segoe UI", 10, "bold"), 
         bg="white", fg="#2c3e50").pack(anchor="w", padx=15, pady=(12, 8))

output_text = scrolledtext.ScrolledText(output_frame, font=("Consolas", 9), 
                                        relief="flat", bg="#ecf0f1", fg="#2c3e50",
                                        wrap="word", height=20)
output_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))
output_text.insert("1.0", "Enter a website URL and click 'Scrape' to extract data.\n\n")
output_text.insert(tk.END, "Example URLs to try:\n")
output_text.insert(tk.END, "  - example.com\n")
output_text.insert(tk.END, "  - python.org\n")
output_text.insert(tk.END, "  - github.com\n\n")
output_text.insert(tk.END, "Note: Install required packages if needed:\n")
output_text.insert(tk.END, "  pip install requests beautifulsoup4")

# Status and buttons
bottom_frame = tk.Frame(content_frame, bg="#f5f6fa")
bottom_frame.pack(fill="x")

status_label = tk.Label(bottom_frame, text="Ready to scrape", 
                       font=("Segoe UI", 9), bg="#f5f6fa", fg="#7f8c8d")
status_label.pack(pady=(0, 10))

buttons_frame = tk.Frame(bottom_frame, bg="#f5f6fa")
buttons_frame.pack()

save_btn = tk.Button(buttons_frame, text="Save to CSV", command=save_data,
                     font=("Segoe UI", 10, "bold"), bg="#27ae60", fg="white",
                     relief="flat", cursor="hand2", padx=20, pady=8,
                     state="disabled")
save_btn.pack(side="left", padx=5)
save_btn.bind("<Enter>", lambda e: save_btn.config(bg="#229954") if save_btn["state"] == "normal" else None)
save_btn.bind("<Leave>", lambda e: save_btn.config(bg="#27ae60") if save_btn["state"] == "normal" else None)

clear_btn = tk.Button(buttons_frame, text="Clear", command=clear_all,
                      font=("Segoe UI", 10, "bold"), bg="#95a5a6", fg="white",
                      relief="flat", cursor="hand2", padx=20, pady=8)
clear_btn.pack(side="left", padx=5)
clear_btn.bind("<Enter>", lambda e: clear_btn.config(bg="#7f8c8d"))
clear_btn.bind("<Leave>", lambda e: clear_btn.config(bg="#95a5a6"))

root.mainloop()
