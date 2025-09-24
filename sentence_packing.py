"""Sentence packing utilities.

Packs individual sentences into sequences up to a token length budget
so that fewer segment boundaries fall mid-sentence. Uses a lightweight
regex-based sentence splitter to avoid heavy dependencies.

Output format (string for each packed example):
  [CLS] Sentence one. [SEP] Sentence two? [SEP] Sentence three! [SEP]

If a single sentence exceeds the max length budget (after accounting for
special tokens) it is chunked into sliding windows (hard split) to avoid
dropping content.
"""
from __future__ import annotations
import re
from typing import List, Iterable, Tuple, Dict, Any

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SPEAKER_RE = re.compile(r"^([A-Z][a-zA-Z]{1,20}):")

FILLER_TERMINALS = {"yeah","mm","erm","uh","er","ah","hmm"}

def simple_sentence_tokenize(text: str) -> List[str]:
    """Very lightweight sentence splitter.
    Splits on punctuation + whitespace boundaries. If no split found, returns [text].
    Trims whitespace. Keeps ending punctuation attached.
    """
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def pack_sentences(lines: Iterable[str], tokenizer, max_length: int, min_fill_ratio: float = 0.85,
                   ensure_terminal_punct: bool = True, max_dangling_tokens: int = 48,
                   allow_borrow: bool = True, borrow_first_sent_max_tokens: int = 8,
                   add_doc_and_speaker: bool = True,
                   collect_stats: bool = False,
                   drop_short_trailing: bool = False,
                   short_trailing_max_tokens: int = 10) -> List[str]:
    """Pack sentences into sequences with a single leading [CLS] and [SEP] after each sentence.

    Args:
        lines: iterable of raw text lines (may contain multiple sentences)
        tokenizer: tokenizer with encode() returning .ids
        max_length: maximum total token length (including specials)
        min_fill_ratio: when finishing a pack, if used length < ratio*max_length,
                        attempt to pull one more sentence if it fits poorly truncated.
    Returns:
        List of packed string samples.
    """
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    doc_token = "[DOC]"
    eod_token = "[EOD]"
    spk_token = "[SPK]"

    # Check tokenizer has the new special tokens; if not, leave them as plain text (they will map to [UNK])
    # This is safe; user can later extend embeddings.
    def has_token(t: str) -> bool:
        try:
            # token_to_id returns None if absent in HF tokenizers; our custom returns id or raises
            return tokenizer.token_to_id(t) is not None
        except Exception:
            return False

    have_doc = has_token(doc_token)
    have_eod = has_token(eod_token)
    have_spk = has_token(spk_token)

    packed = []
    stats: Dict[str, Any] = {
        "packs": 0,
        "dangling_carried": 0,
        "borrow_moves": 0,
        "nonterminal_end": 0,
        "terminal_end": 0,
        "short_trailing_dropped": 0,
    } if collect_stats else None
    current_sentences: List[str] = []
    current_len = 1  # account for [CLS]

    def is_sentence_terminal(sentence: str) -> bool:
        if not sentence:
            return False
        if sentence[-1] in '.!?':
            return True
        low = sentence.lower().strip("'\"")
        if low in FILLER_TERMINALS:
            return True
        return False

    def flush(force: bool=False):
        nonlocal current_sentences, current_len
        if not current_sentences:
            return
        # If we want punctuation-final endings, attempt to carry over a dangling sentence.
        if ensure_terminal_punct and not force:
            # If last sentence lacks terminal punctuation and is short, carry it to next pack.
            last = current_sentences[-1]
            if (last and not is_sentence_terminal(last)):
                # Estimate token count (rough); if short, pop and delay.
                last_ids = tokenizer.encode(last).ids
                if len(last_ids) <= max_dangling_tokens:
                    if stats is not None:
                        stats["dangling_carried"] += 1
                    dangling = current_sentences.pop()
                    # Only flush remaining if any remain.
                    if current_sentences:
                        sample = " ".join([cls_token] + [s + f" {sep_token}" for s in current_sentences])
                        packed.append(sample)
                        if stats is not None:
                            last_sent = current_sentences[-1]
                            if is_sentence_terminal(last_sent):
                                stats["terminal_end"] += 1
                            else:
                                stats["nonterminal_end"] += 1
                            stats["packs"] += 1
                    # Start next pack with dangling sentence.
                    current_sentences = [dangling]
                    current_len = 1 + len(tokenizer.encode(dangling).ids) + 1
                    return
        # Normal flush
        if drop_short_trailing and not force and current_sentences:
            last = current_sentences[-1]
            if not is_sentence_terminal(last):
                last_ids = tokenizer.encode(last).ids
                fill_ratio = current_len / max_length
                if len(last_ids) <= short_trailing_max_tokens and fill_ratio >= min_fill_ratio:
                    # Drop last and carry forward
                    current_sentences.pop()
                    if stats is not None:
                        stats["short_trailing_dropped"] += 1
                    if current_sentences:
                        sample = " ".join([cls_token] + [s + f" {sep_token}" for s in current_sentences])
                        packed.append(sample)
                        if stats is not None:
                            last_sent = current_sentences[-1]
                            if is_sentence_terminal(last_sent):
                                stats["terminal_end"] += 1
                            else:
                                stats["nonterminal_end"] += 1
                            stats["packs"] += 1
                    # Start new pack buffer with dropped fragment
                    current_sentences = [last]
                    current_len = 1 + len(last_ids) + 1
                    return
        sample = " ".join([cls_token] + [s + f" {sep_token}" for s in current_sentences])
        packed.append(sample)
        if stats is not None:
            last_sent = current_sentences[-1]
            if is_sentence_terminal(last_sent):
                stats["terminal_end"] += 1
            else:
                stats["nonterminal_end"] += 1
            stats["packs"] += 1
        current_sentences = []
        current_len = 1

    # Preprocess to inject doc boundaries and speaker markers if requested
    processed_lines: List[str] = []
    if add_doc_and_speaker:
        # Detect document boundaries: lines list may contain explicit markers [DOC]/[EOD]
        # We assume upstream will have inserted them if desired; if not present we just treat all lines as one doc.
        in_doc = False
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line == doc_token:
                in_doc = True
                processed_lines.append(line if have_doc else line)
                continue
            if line == eod_token:
                processed_lines.append(line if have_eod else line)
                in_doc = False
                continue
            # Speaker tagging heuristic
            if SPEAKER_RE.match(line) and have_spk:
                line = f"{spk_token} {line}"
            processed_lines.append(line)
    else:
        processed_lines = [l for l in lines if l.strip()]

    for line in processed_lines:
        # Split line into sentences
        sentences = simple_sentence_tokenize(line)
        if not sentences:
            continue
        for sent in sentences:
            # token length for this sentence (excluding specials) + one SEP
            sent_ids = tokenizer.encode(sent).ids
            sent_len = len(sent_ids) + 1  # +1 for [SEP]
            # If this single sentence is too long to ever fit (cls + sent + sep > max), chunk it
            budget = max_length - 2  # reserve 2 for [CLS] + final [SEP]
            if sent_len + 1 > max_length:  # > max_length including cls already counted elsewhere
                # Flush what we have
                flush()
                # Hard chunk sentence tokens
                tokens = sent_ids
                window = budget
                start = 0
                while start < len(tokens):
                    piece_ids = tokens[start:start+window]
                    piece_text = tokenizer.decode(piece_ids).strip()
                    packed.append(f"{cls_token} {piece_text} {sep_token}")
                    start += window
                continue

            # Normal case: attempt to add sentence to current pack
            prospective = current_len + sent_len
            if prospective > max_length:
                # Decide whether to flush or try to fill more (if current pack badly under-filled) — we flush now
                flush()
                current_sentences.append(sent)
                current_len = 1 + sent_len
            else:
                current_sentences.append(sent)
                current_len = prospective

        # After processing a line, if we are far from filling, continue; else consider flush opportunistically
        if current_len / max_length >= min_fill_ratio:
            flush()

    # Flush remainder
    flush(force=True)

    # Backward borrow pass: attempt to move a short first sentence from a pack to previous
    if allow_borrow and len(packed) > 1:
        adjusted: List[str] = []
        prev_sent_lists: List[List[str]] = []
        # Recover sentence lists from packed strings (lossy if sentences had inner [SEP], acceptable here)
        def unpack(sample: str) -> List[str]:
            # sample pattern: [CLS] S1 [SEP] S2 [SEP] ...
            parts = sample.split()
            # Reconstruct by splitting on [SEP]
            collected = " ".join(parts[1:])  # drop [CLS]
            raw_sents = [s.strip() for s in collected.split(sep_token) if s.strip()]
            return raw_sents
        sent_lists = [unpack(p) for p in packed]
        changed = False
        for i in range(1, len(sent_lists)):
            prev_list = sent_lists[i-1]
            curr_list = sent_lists[i]
            if not prev_list or not curr_list:
                continue
            last_prev = prev_list[-1]
            first_curr = curr_list[0]
            if not is_sentence_terminal(last_prev) and is_sentence_terminal(first_curr):
                # token count check
                prev_ids = []
                for s in prev_list:
                    prev_ids.extend(tokenizer.encode(s).ids + [tokenizer.token_to_id(sep_token)])
                first_ids = tokenizer.encode(first_curr).ids
                projected = 1 + len(prev_ids) + len(first_ids) + 1  # [CLS] + prev + first + final [SEP]
                if projected <= max_length and len(first_ids) <= borrow_first_sent_max_tokens:
                    # move
                    prev_list.append(first_curr)
                    del curr_list[0]
                    changed = True
                    if stats is not None:
                        stats["borrow_moves"] += 1
        if changed:
            # Rebuild packed strings
            rebuilt = []
            for sl in sent_lists:
                if not sl:
                    continue
                rebuilt.append(" ".join([cls_token] + [s + f" {sep_token}" for s in sl]))
            packed = rebuilt

    # Final strict length enforcement
    final_packed = []
    for sample in packed:
        ids = tokenizer.encode(sample.replace(cls_token, '').replace(sep_token, '')).ids
        if len(ids) <= max_length:
            final_packed.append(sample)
        else:
            # Attempt to drop last sentence and retry
            sents = sample.split(f" {sep_token}")
            sents = [s for s in sents if s.strip()]
            if len(sents) > 1:
                trimmed = sents[:-1]
                new_sample = " ".join([cls_token] + [s.strip() + f" {sep_token}" for s in trimmed])
                ids2 = tokenizer.encode(new_sample.replace(cls_token,'').replace(sep_token,'')).ids
                if len(ids2) <= max_length:
                    final_packed.append(new_sample)
                else:
                    # Drop entirely if still too long (rare)
                    continue
            else:
                continue
    if stats is not None:
        return final_packed, stats
    return final_packed

__all__ = ["pack_sentences", "simple_sentence_tokenize"]
