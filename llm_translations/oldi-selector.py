#!/usr/bin/env python3
"""
Interactive selector for 100 sentence pairs from Dataset/oldi/not_normalized/pairs_all.tsv

Behavior:
- Shows Italian (col 0), then English (col 1), then other columns if present.
- Prompt: y = accept, n = reject, q = quit and save progress.
- Shows sentences in random order, each only once. Shows a counter of how many possible sentences remain.
- Saves accepted sentences to oldi_selected.tsv as they are accepted and tracks seen items so you can continue later.
"""
import os
import sys
import json
import random
import tkinter as tk
from tkinter import messagebox


DATA_PATH = os.path.join('Dataset', 'oldi', 'not_normalized', 'pairs_all.tsv')
SELECTED_PATH = 'llm-translations/oldi_selected.tsv'
SEEN_PATH = 'llm-translations/oldi_seen.json'
TARGET = 100


def load_lines(path):
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]
    return lines


def load_seen(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_seen(path, seen):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(seen), f, ensure_ascii=False)


def append_selected(path, line):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def show_item(line):
    parts = line.split('\t')
    # column layout: id, italiano, veneto, siciliano, lombardo, sardo, ligure, friulano, inglese, spagnolo, francese, tedesco, catalano, sloveno
    labels = ['id', 'italiano', 'veneto', 'siciliano', 'lombardo', 'sardo', 'ligure', 'friulano', 'inglese', 'spagnolo', 'francese', 'tedesco', 'catalano', 'sloveno']
    # ensure parts list is as long as labels
    while len(parts) < len(labels):
        parts.append('')
    it = parts[1]
    en = parts[8]
    print('\nItalian:')
    print(it)
    print('\nEnglish:')
    print(en)
    # print other languages (excluding id, italiano, inglese)
    others_lines = []
    for idx, lab in enumerate(labels):
        if lab in ('id', 'italiano', 'inglese'):
            continue
        val = parts[idx] if idx < len(parts) else ''
        if val:
            others_lines.append(f"{lab}: {val}")
    if others_lines:
        print('\nOther languages:')
        print('\n'.join(others_lines))


def make_gui(lines, seen, selected_count):
    remaining_indices = [i for i in range(len(lines)) if str(i) not in seen]
    random.shuffle(remaining_indices)

    root = tk.Tk()
    root.title('OLDI Selector')
    root.geometry('900x600')

    import tkinter.font as tkfont

    counter_var = tk.StringVar()
    current = {'idx': None}

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill='both', expand=True)

    # Fonts
    header_font = tkfont.Font(family='Helvetica', size=14, weight='bold')
    sent_font = tkfont.Font(family='Helvetica', size=12)
    other_font = tkfont.Font(family='Helvetica', size=10, slant='italic')

    counter_lbl = tk.Label(frame, textvariable=counter_var, anchor='w', fg='#333333')
    counter_lbl.pack(fill='x')

    # Italian and English panes with clear labels
    top = tk.Frame(frame)
    top.pack(fill='both', expand=False, pady=(8, 8))

    ital_frame = tk.Frame(top)
    ital_frame.pack(side='left', fill='both', expand=True, padx=(0,6))
    tk.Label(ital_frame, text='Italian', font=header_font, fg='#1b5e20').pack(anchor='w')
    italian_txt = tk.Text(ital_frame, wrap='word', height=6, font=sent_font, bd=1, relief='solid')
    italian_txt.pack(fill='both', expand=True, pady=(4,0))
    italian_txt.configure(state='disabled', bg='#fbfff9')

    eng_frame = tk.Frame(top)
    eng_frame.pack(side='left', fill='both', expand=True, padx=(6,0))
    tk.Label(eng_frame, text='English', font=header_font, fg='#0d47a1').pack(anchor='w')
    english_txt = tk.Text(eng_frame, wrap='word', height=6, font=sent_font, bd=1, relief='solid')
    english_txt.pack(fill='both', expand=True, pady=(4,0))
    english_txt.configure(state='disabled', bg='#f7fbff')

    # Other languages area
    other_frame = tk.Frame(frame)
    other_frame.pack(fill='both', expand=True)
    other_header_font = tkfont.Font(family='Helvetica', size=10, weight='bold', slant='italic')
    tk.Label(other_frame, text='Other languages', font=other_header_font, fg='#555555').pack(anchor='w')
    other_txt = tk.Text(other_frame, wrap='word', height=8, font=other_font, bd=1, relief='solid')
    other_txt.pack(fill='both', expand=True, pady=(4,0))
    other_txt.configure(state='disabled', bg='#ffffff')

    def refresh_counter():
        remaining = len([i for i in range(len(lines)) if str(i) not in seen])
        counter_var.set(f"Remaining possible sentences: {remaining}  (need {TARGET - selected_count['count']} more) | Press 'y' to accept, 'n' to reject, 'q' to quit")

    def render(idx):
        line = lines[idx]
        parts = line.split('\t')
        labels = ['id', 'italiano', 'veneto', 'siciliano', 'lombardo', 'sardo', 'ligure', 'friulano', 'inglese', 'spagnolo', 'francese', 'tedesco', 'catalano', 'sloveno']
        while len(parts) < len(labels):
            parts.append('')
        it = parts[1]
        en = parts[8]
        italian_txt.configure(state='normal')
        italian_txt.delete('1.0', tk.END)
        italian_txt.insert(tk.END, it)
        italian_txt.configure(state='disabled')

        english_txt.configure(state='normal')
        english_txt.delete('1.0', tk.END)
        english_txt.insert(tk.END, en)
        english_txt.configure(state='disabled')

        other_txt.configure(state='normal')
        other_txt.delete('1.0', tk.END)
        other_lines = []
        for idx2, lab in enumerate(labels):
            if lab in ('id', 'italiano', 'inglese'):
                continue
            val = parts[idx2]
            if val:
                other_lines.append(f"{lab}: {val}")
        if other_lines:
            other_txt.insert(tk.END, "\n".join(other_lines))
        other_txt.configure(state='disabled')

    def next_item():
        refresh_counter()
        if selected_count['count'] >= TARGET:
            messagebox.showinfo('Done', f"Finished: selected {selected_count['count']} sentences.")
            root.destroy()
            return
        if not remaining_indices:
            messagebox.showinfo('Done', 'No more unseen sentences.')
            root.destroy()
            return
        current['idx'] = remaining_indices.pop(0)
        render(current['idx'])

    def accept():
        idx = current['idx']
        if idx is None:
            return
        append_selected(SELECTED_PATH, lines[idx])
        selected_count['count'] += 1
        seen.add(str(idx))
        save_seen(SEEN_PATH, seen)
        next_item()

    def reject():
        idx = current['idx']
        if idx is None:
            return
        seen.add(str(idx))
        save_seen(SEEN_PATH, seen)
        next_item()

    def quit_app():
        save_seen(SEEN_PATH, seen)
        root.destroy()

    def on_key(event):
        key = event.char.lower()
        if key == 'y':
            accept()
        elif key == 'n':
            reject()
        elif key == 'q':
            quit_app()

    root.bind('<Key>', on_key)
    root.protocol('WM_DELETE_WINDOW', quit_app)
    next_item()
    root.mainloop()


def main():
    lines = load_lines(DATA_PATH)
    seen = load_seen(SEEN_PATH)
    selected_count = {'count': 0}
    if os.path.exists(SELECTED_PATH):
        # count already selected
        with open(SELECTED_PATH, 'r', encoding='utf-8') as f:
            selected_count['count'] = sum(1 for _ in f)

    if selected_count['count'] >= TARGET:
        print(f"Already have {selected_count['count']} selected. Finished.")
        return
    make_gui(lines, seen, selected_count)


if __name__ == '__main__':
    main()
