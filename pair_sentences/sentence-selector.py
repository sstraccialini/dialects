#!/usr/bin/env python3
"""
Interactive selector for 100 sentence pairs from Dataset/oldi/not_normalized/pairs_all.tsv

Behavior:
- Shows Italian (col 0), then English (col 1), then other columns if present.
- Prompt: y = accept, n = reject, q = quit and save progress.
- Shows sentences in random order, each only once. Shows a counter of how many possible sentences remain.
- Saves accepted sentences to pair_sentences/selected.tsv as they are accepted and tracks seen items so you can continue later.
"""
import os
import sys
import json
import random
import re
import tkinter as tk
from tkinter import messagebox

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'merged.tsv')
SELECTED_PATH = os.path.join(CURRENT_DIR, 'selected.tsv')
SEEN_PATH = os.path.join(CURRENT_DIR, 'already_classified.json')

def load_lines(path):
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [re.sub(r'["”]+', '"', ln.rstrip('\n')) for ln in f]
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


def remove_last_selected(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
    if not rows:
        return
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(rows[:-1])


def show_item(line):
    parts = line.split('\t')
    # column layout: unique_id, dataset, original_id, italiano, veneto, siciliano, lombardo, sardo, ligure, friulano, inglese, spagnolo, francese, tedesco, catalano, sloveno
    labels = ['unique_id', 'dataset', 'original_id', 'italiano', 'veneto', 'siciliano', 'lombardo', 'sardo', 'ligure', 'friulano', 'inglese', 'spagnolo', 'francese', 'tedesco', 'catalano', 'sloveno']
    # ensure parts list is as long as labels
    while len(parts) < len(labels):
        parts.append('')
    it = parts[3]
    en = parts[10]
    print(f"\nInfo: ID={parts[0]}, Dataset={parts[1]}, Original ID={parts[2]}")
    print('\nItalian:')
    print(it)
    print('\nEnglish:')
    print(en)
    # print other languages
    others_lines = []
    for idx, lab in enumerate(labels):
        if lab in ('unique_id', 'dataset', 'original_id', 'italiano', 'inglese'):
            continue
        val = parts[idx] if idx < len(parts) else ''
        if val:
            others_lines.append(f"{lab}: {val}")
    if others_lines:
        print('\nOther languages:')
        print('\n'.join(others_lines))


def make_gui(lines, seen, selected_count, start_idx, end_idx):
    # select indices from preselected range and not already classified
    remaining_indices = [i for i in range(start_idx, end_idx + 1) if 0 <= i < len(lines) and str(i) not in seen]
    random.shuffle(remaining_indices)

    root = tk.Tk()
    root.title('OLDI Selector')
    root.geometry('900x600')

    import tkinter.font as tkfont

    counter_var = tk.StringVar()
    info_var = tk.StringVar()
    current = {'idx': None}
    history = []

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill='both', expand=True)

    # Fonts
    header_font = tkfont.Font(family='Helvetica', size=14, weight='bold')
    sent_font = tkfont.Font(family='Helvetica', size=12)
    other_font = tkfont.Font(family='Helvetica', size=11, slant='italic')

    counter_lbl = tk.Label(frame, textvariable=counter_var, anchor='w', fg='#333333', font=(None, 12))
    counter_lbl.pack(fill='x')
    info_lbl = tk.Label(frame, textvariable=info_var, anchor='w', fg='#555555', font=(None, 11, 'bold'))
    info_lbl.pack(fill='x', pady=(0, 8))

    # Two boxes, one below the other
    ital_frame = tk.Frame(frame)
    ital_frame.pack(fill='both', expand=True, pady=(0, 8))
    tk.Label(ital_frame, text='Italian', font=(header_font.actual('family'), 16, 'bold'), fg='#1b5e20').pack(anchor='w')
    italian_txt = tk.Text(ital_frame, wrap='word', height=6, font=(sent_font.actual('family'), 14), bd=2, relief='solid')
    italian_txt.pack(fill='both', expand=True, pady=(2,0))
    italian_txt.configure(state='disabled', bg='#fbfff9')

    other_frame = tk.Frame(frame)
    other_frame.pack(fill='both', expand=True)
    tk.Label(other_frame, text='Other languages', font=(header_font.actual('family'), 14, 'bold'), fg='#0d47a1').pack(anchor='w')
    other_txt = tk.Text(other_frame, wrap='word', height=10, font=other_font, bd=2, relief='solid')
    other_txt.pack(fill='both', expand=True, pady=(2,0))
    other_txt.configure(state='disabled', bg='#f7fbff')

    def refresh_counter():
        remaining = len([i for i in range(start_idx, end_idx + 1) if 0 <= i < len(lines) and str(i) not in seen])
        counter_var.set(f"Remaining possible sentences: {remaining} | y = accept, n = reject, z = undo, q = quit")

    def render(idx):
        line = lines[idx]
        parts = line.split('\t')
        labels = ['unique_id', 'dataset', 'original_id', 'italiano', 'veneto', 'siciliano', 'lombardo', 'sardo', 'ligure', 'friulano', 'inglese', 'spagnolo', 'francese', 'tedesco', 'catalano', 'sloveno']
        while len(parts) < len(labels):
            parts.append('')
        it = parts[3]
        en = parts[10]

        info_var.set(f"[ID: {parts[0]} | Dataset: {parts[1]} | Orig ID: {parts[2]}]")

        italian_txt.configure(state='normal')
        italian_txt.delete('1.0', tk.END)
        italian_txt.insert(tk.END, it)
        italian_txt.configure(state='disabled')

        other_txt.configure(state='normal')
        other_txt.delete('1.0', tk.END)
        
        lines_out = []
        if en:
            lines_out.append(f"English: {en}")
        for idx2, lab in enumerate(labels):
            if lab in ('unique_id', 'dataset', 'original_id', 'italiano', 'inglese'):
                continue
            val = parts[idx2]
            if val:
                lines_out.append(f"{lab}: {val}")
        
        if lines_out:
            other_txt.insert(tk.END, "\n".join(lines_out))
        other_txt.configure(state='disabled')

    def next_item():
        refresh_counter()
        if not remaining_indices:
            messagebox.showinfo('Done', f"Finished: selected {selected_count['count']} sentences.")
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
        history.append(('accept', idx))
        next_item()

    def reject():
        idx = current['idx']
        if idx is None:
            return
        seen.add(str(idx))
        save_seen(SEEN_PATH, seen)
        history.append(('reject', idx))
        next_item()

    def undo_last():
        if not history:
            return
        action, idx = history.pop()
        seen.discard(str(idx))
        if action == 'accept':
            remove_last_selected(SELECTED_PATH)
            selected_count['count'] = max(0, selected_count['count'] - 1)
        save_seen(SEEN_PATH, seen)
        remaining_indices.append(idx)
        random.shuffle(remaining_indices)
        current['idx'] = None
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
        elif key == 'z':
            undo_last()
        elif key == 'q':
            quit_app()

    root.bind('<Key>', on_key)
    root.protocol('WM_DELETE_WINDOW', quit_app)
    # Big accept/reject buttons
    btn_frame = tk.Frame(frame, pady=8)
    btn_frame.pack(fill='x')
    accept_btn = tk.Button(btn_frame, text="Y = Accept", bg='#2e7d32', fg='white', font=(None, 14, 'bold'), command=accept)
    accept_btn.pack(side='left', expand=True, fill='x', padx=6)
    reject_btn = tk.Button(btn_frame, text="N = Reject", bg='#c62828', fg='white', font=(None, 14, 'bold'), command=reject)
    reject_btn.pack(side='left', expand=True, fill='x', padx=6)
    undo_btn = tk.Button(btn_frame, text="Z = Undo", bg='#6a1b9a', fg='white', font=(None, 14, 'bold'), command=undo_last)
    undo_btn.pack(side='left', expand=True, fill='x', padx=6)
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

    make_gui(lines, seen, selected_count, 0, len(lines) - 1)


if __name__ == '__main__':
    main()
