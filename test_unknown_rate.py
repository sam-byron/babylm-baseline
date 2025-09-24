from tokenizers import Tokenizer
import glob, re

tok = Tokenizer.from_file("tokenizer.json")
unk_id = tok.token_to_id("[UNK]")
placeholder_pattern = re.compile(r"\\[UNK\\]")

true_unk = 0
placeholders = 0
total = 0

for fp in glob.glob("data/pretrain/bnc/**/*.md", recursive=True)[:500]:  # sample subset
    for line in open(fp, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        ph = len(placeholder_pattern.findall(line))
        if ph:
            safe_line = line.replace("[UNK]", "⟪UNK⟫")
        else:
            safe_line = line
        enc = tok.encode(safe_line)
        ids = enc.ids
        total += len(ids)
        placeholders += ph  # counts number of placeholder tokens (will later be 1 per occurrence)
        true_unk += sum(1 for i in ids if i == unk_id)

print("Total tokens:", total, "Raw placeholder [UNK] occurrences:", placeholders, "Tokenizer fallback unknowns:", true_unk)