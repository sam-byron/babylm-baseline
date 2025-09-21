import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import gzip


def _load_chunk_sentences(chunk_path):
    """Helper function for multiprocessing chunk loading - robust version with memory management."""
    try:
        import torch
        import os
        import gc
        
        # Get file info
        filename = os.path.basename(chunk_path)
        file_size = os.path.getsize(chunk_path)
        
        # Load chunk data with memory optimization
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        sentences = []
        
        # Process sentences efficiently
        if isinstance(chunk_data, list):
            for sentence_tokens in chunk_data:
                if isinstance(sentence_tokens, (list, torch.Tensor)) and len(sentence_tokens) > 0:
                    # Convert to tensor if needed
                    if not isinstance(sentence_tokens, torch.Tensor):
                        sentence_tokens = torch.tensor(sentence_tokens, dtype=torch.long)
                    elif sentence_tokens.dtype != torch.long:
                        sentence_tokens = sentence_tokens.long()
                    
                    # Basic validation
                    if len(sentence_tokens) > 0:
                        sentences.append(sentence_tokens)
        
        # Aggressive cleanup to prevent memory leaks
        del chunk_data
        gc.collect()
        
        # Progress indicator
        print(f"✓ {filename}: {len(sentences)} sentences ({file_size/(1024*1024):.1f}MB)")
        
        return sentences
        
    except Exception as e:
        print(f"❌ Error loading {chunk_path}: {e}")
        # Force cleanup on error
        import gc
        gc.collect()
        return []  # Return empty list instead of None for easier handling


class Indexer:
    def __init__(self, documents):
        lengths = [len(document) for document in documents]
        self.cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)

    def get_indices(self, index):
        document_index = torch.searchsorted(self.cumsum, index, right=True).item() - 1
        segment_index = index - self.cumsum[document_index]
        return document_index, segment_index

    def __len__(self):
        return self.cumsum[-1].item()


class AbstractMaskingStrategy:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1, max_span_length=10):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.max_span_length = max_span_length
        self.mask_index = self.tokenizer.token_to_id("[MASK]")

    def __call__(self, tokens):
        raise NotImplementedError()


class SubwordMaskingStrategy(AbstractMaskingStrategy):
    def __call__(self, tokens):
        labels = tokens.clone()

        probability_matrix = torch.full(tokens.shape, self.mask_p)
        special_tokens_mask = labels < self.n_special_tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.padding_label_id  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0 - self.random_p - self.keep_p)).bool() & masked_indices
        tokens[indices_replaced] = self.mask_index

        # 10% of the time, we replace masked input tokens with random word
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        indices_random = torch.bernoulli(torch.full(labels.shape, self.random_p / (self.random_p + self.keep_p + 1e-6))).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            low=self.n_special_tokens - 1,
            high=self.tokenizer.get_vocab_size(),
            size=(indices_random.sum(),),
            dtype=torch.long
        )
        tokens[indices_random] = random_words

        return tokens, labels


class SpanMaskingStrategy(AbstractMaskingStrategy):
    def __call__(self, tokens):
        # Print max_span_length for debugging
        # print(f"SpanMaskingStrategy called with max_span_length={self.max_span_length}") 
        labels = torch.full_like(tokens, fill_value=self.padding_label_id)

        n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
        preservation_mask = tokens < self.n_special_tokens
        while n_masked > 0:
            span_length = torch.tensor([0]).geometric_(1/3).item() % self.max_span_length
            offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            mask[preservation_mask] = False

            labels = torch.where(mask, tokens, labels)
            random_p = torch.rand([]).item()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            if random_p < 1.0 - self.random_p - self.keep_p:
                tokens[mask] = self.mask_index
            elif random_p < 1.0 - self.keep_p:
                random_words = torch.randint(
                    low=self.n_special_tokens - 1,
                    high=self.tokenizer.get_vocab_size(),
                    size=(mask.sum(),),
                    dtype=torch.long
                )
                tokens[mask] = random_words

            n_masked -= mask.sum()

        return tokens, labels


class WholeWordMaskingStrategy(AbstractMaskingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.continuous_set = set(
            token_id
            for token_id in range(self.tokenizer.get_vocab_size())
            if self.tokenizer.id_to_token(token_id).startswith("##")  # Changed from "@@" to "##"
        )

    def __call__(self, tokens):
        labels = torch.full_like(tokens, fill_value=self.padding_label_id)

        words, next_word = [], None
        for i, token_id in enumerate(tokens.tolist()):
            if token_id in self.continuous_set:
                if next_word is not None:
                    next_word[1] += 1
            else:
                if next_word is not None:
                    words.append(next_word)

                if token_id < self.n_special_tokens:
                    next_word = None
                else:
                    next_word = [i, i+1]

        if next_word is not None:
            words.append(next_word)

        mask = torch.bernoulli(torch.full([len(words)], self.mask_p))
        for span_id in mask.nonzero().squeeze(-1).tolist():
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            mask[words[span_id][0] : words[span_id][1]] = True

            labels = torch.where(mask, tokens, labels)
            random_p = torch.rand([]).item()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            if random_p < 1.0 - self.random_p - self.keep_p:
                tokens[mask] = self.mask_index
            elif random_p < 1.0 - self.keep_p:
                random_words = torch.randint(
                    low=self.n_special_tokens - 1,
                    high=self.tokenizer.get_vocab_size(),
                    size=(mask.sum(),),
                    dtype=torch.long
                )
                tokens[mask] = random_words

        return tokens, labels


class AbstractMlmDataset(Dataset):
    def __init__(self, input_file, tokenizer, masking_strategy: str, seq_length=128, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = {
            "subword": SubwordMaskingStrategy,
            "span": SpanMaskingStrategy,
            "whole_word": WholeWordMaskingStrategy,
        }[masking_strategy]
        self.masking_strategy = self.masking_strategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = torch.tensor([self.tokenizer.token_to_id("[CLS]")], dtype=torch.int16)
        self.sep_index = torch.tensor([self.tokenizer.token_to_id("[SEP]")], dtype=torch.int16)
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        # every document contains a list of sentences, which are themselves np arrays of integers
        with gzip.open(input_file, "rb") as f:
            self.documents = pickle.load(f)

        self.documents = [
            [torch.from_numpy(sentence.astype(np.int16)) for sentence in document]
            for document in self.documents
        ]
        self.documents = [self.create_segments(document) for document in self.documents]
        self.documents = [document for document in self.documents if len(document) > 0]
        self.indexer = Indexer(self.documents)

    def __len__(self):
        return len(self.indexer)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    # turn sentences into tuples of (sentence, offset)
    # segment + the next $offset sentences forms one segment
    def create_segments(self, document):
        segments = []
        for i_sentence, sentence in enumerate(document):
            length = len(sentence)
            if length == 0:
                continue
            segment = [sentence]
            offsets = [length]
            offset = i_sentence
            while length < self.seq_length - 3 and offset + 1 < len(document):
                offset += 1
                if len(document[offset]) == 0:
                    continue
                segment.append(document[offset])
                length += len(document[offset])
                offsets.append(length)

            segments.append((torch.cat(segment), offsets))

        return segments

    def __getitem__(self, index):
        tokens, attention_mask, is_next_sentence = self.get_segment(index)
        inputs, outputs = self.mask_tokens(tokens)
        return inputs, attention_mask, outputs, is_next_sentence

    def get_segment(self, index):
        pass

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            return tokens_a, tokens_b
        cut = total_length - max_num_tokens
        cut_left = self.randint(max(0, cut - len(tokens_b) + 1), min(cut + 1, len(tokens_a)))
        cut_right = cut - cut_left
        tokens_a = tokens_a[:len(tokens_a) - cut_left]
        tokens_b = tokens_b[:len(tokens_b) - cut_right]
        return tokens_a, tokens_b

    def mask_tokens(self, tokens):
        return self.masking_strategy(tokens)


class DocumentDataset(AbstractMlmDataset):
    def get_segment(self, index):
        document_index, segment_index = self.indexer.get_indices(index)
        segment, segment_offsets = self.documents[document_index][segment_index]
        is_next_sentence = self.rand() > 0.5 and len(segment_offsets) > 1

        # different next sentence
        if not is_next_sentence:

            # try to find a random -- and different -- document
            # should succeed in the first loop, but just in case...
            for _ in range(10):
                document_b_index, segment_index = self.indexer.get_indices(self.randint(0, len(self)))
                if document_b_index != document_index:
                    break

            # if we were unlucky and didn't find a different article
            if document_b_index == document_index:
                is_next_sentence = True

            segment_a = segment
            segment_b, _ = self.documents[document_b_index][segment_index]

        if is_next_sentence:
            a_end = self.randint(0, len(segment_offsets) - 1)
            segment_a = segment[:segment_offsets[a_end]]
            segment_b = segment[segment_offsets[a_end]:]

        target_seq_length = self.seq_length - 3 if self.rand() > self.short_p else self.randint(2, self.seq_length - 3)
        segment_a, segment_b = self.truncate_seq_pair(segment_a, segment_b, target_seq_length)

        assert len(segment_a) >= 1
        assert len(segment_b) >= 1
        assert len(segment_a) + len(segment_b) <= target_seq_length

        padding_length = self.seq_length - (len(segment_a) + len(segment_b) + 3)
        padding = torch.full((padding_length,), fill_value=self.pad_index, dtype=torch.int16)
        segment = [self.cls_index, segment_a, self.sep_index, segment_b, self.sep_index, padding]
        segment = torch.cat(segment).long()

        attention_mask = torch.cat([
            torch.zeros(len(segment_a) + len(segment_b) + 3, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        is_next_sentence = torch.tensor(is_next_sentence, dtype=torch.long)

        return segment, attention_mask, is_next_sentence


class OrderDataset(AbstractMlmDataset):
    def get_segment(self, index):
        document_index, segment_index = self.indexer.get_indices(index)
        segment, segment_offsets = self.documents[document_index][segment_index]

        if len(segment_offsets) == 1:
            target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
            segment_a = segment[:target_seq_length]
            padding_length = self.seq_length - (len(segment_a) + 2)
            padding = torch.full((padding_length,), fill_value=self.pad_index, dtype=torch.int16)
            segment = torch.cat([self.cls_index, segment_a, self.sep_index, padding]).long()
            attention_mask = torch.cat([
                torch.zeros(len(segment_a) + 2, dtype=torch.bool),
                torch.ones(padding_length, dtype=torch.bool)
            ])
            is_next_sentence = torch.tensor(False, dtype=torch.long)
            return segment, attention_mask, is_next_sentence

        a_end = self.randint(0, len(segment_offsets) - 1)
        segment_a = segment[:segment_offsets[a_end]]
        segment_b = segment[segment_offsets[a_end]:]

        is_next_sentence = self.rand() > 0.5
        if not is_next_sentence:
            segment_a, segment_b = segment_b, segment_a

        target_seq_length = self.seq_length - 3 if self.rand() > self.short_p else self.randint(2, self.seq_length - 3)
        segment_a, segment_b = self.truncate_seq_pair(segment_a, segment_b, target_seq_length)

        assert len(segment_a) >= 1
        assert len(segment_b) >= 1
        assert len(segment_a) + len(segment_b) <= target_seq_length

        padding_length = self.seq_length - (len(segment_a) + len(segment_b) + 3)
        padding = torch.full((padding_length,), fill_value=self.pad_index, dtype=torch.int16)
        segment = [self.cls_index, segment_a, self.sep_index, segment_b, self.sep_index, padding]
        segment = torch.cat(segment).long()

        attention_mask = torch.cat([
            torch.zeros(len(segment_a) + len(segment_b) + 3, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])
        is_next_sentence = torch.tensor(is_next_sentence, dtype=torch.long)
        return segment, attention_mask, is_next_sentence


class SentenceAwareDataset(Dataset):
    """Dataset that loads tokenized sentences with preserved boundaries from PT files."""
    
    def __init__(self, cache_path, tokenizer, seq_length=512, mask_p=0.15, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.n_special_tokens = 6
        self.padding_label_id = -100
        
        # Initialize token IDs
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        
        # Load all chunk files with robust multiprocessing
        import glob
        import os
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp
        from functools import partial
        
        chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
        print(f"Loading {len(chunk_paths)} chunk files for sentence-aware MLM training...")
        
        if len(chunk_paths) == 0:
            raise ValueError(f"No chunk files found in {cache_path}")
        
        self.sentences = []
        
        # Use simple sequential loading for reliability
        print(f"💾 Memory before loading: {self._get_memory_usage()}")
        
        for i, path in enumerate(chunk_paths):
            try:
                result = _load_chunk_sentences(path)
                if result:
                    self.sentences.extend(result)
                if (i + 1) % 10 == 0:
                    print(f"Sequential: {i + 1}/{len(chunk_paths)} chunks loaded, {len(self.sentences)} sentences so far")
                    print(f"💾 Memory usage: {self._get_memory_usage()}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"✅ Loaded {len(self.sentences)} sentences from {len(chunk_paths)} chunks")
        
        # Create multi-sentence sequences first
        try:
            self.sequences = self._create_multi_sentence_sequences()
            print(f"✅ Successfully created {len(self.sequences)} sequences")
        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            # Fallback: create simple sequences from individual sentences
            print("Creating fallback individual sentence sequences...")
            self.sequences = self.sentences[:10000]  # Use first 10k sentences as sequences
            print(f"✅ Created {len(self.sequences)} fallback sequences")
        
        # EMERGENCY: Clean up sentences list to save memory AFTER sequence creation
        print("🧹 Cleaning up intermediate sentences data to save memory...")
        temp_sequences_count = len(self.sequences)
        del self.sentences  # Free memory from sentences list
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        print(f"✅ Memory cleanup complete. Dataset ready with {temp_sequences_count} sequences")
        
    def _get_memory_usage(self):
        """Get current memory usage in a readable format."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except:
            return "unknown"
    
    def _create_multi_sentence_sequences(self):
        """Combine multiple sentences into sequences up to seq_length tokens."""
        print("Creating multi-sentence sequences for long-range learning...")
        sequences = []
        current_sequence = []
        current_length = 0
        
        # Pre-allocate pad tensor for efficiency
        pad_tensor_cache = {}
        
        for i, sentence in enumerate(self.sentences):
            sentence_length = len(sentence)
            
            # Truncate long sentences instead of skipping them - preserve all data!
            if sentence_length > self.seq_length:
                if i % 10000 == 0:  # Only log every 10,000th truncation to reduce spam
                    print(f"Truncating sentence {i} from length {sentence_length} to {self.seq_length}")
                sentence = sentence[:self.seq_length]  # Truncate to max length
                sentence_length = self.seq_length
            
            # If adding this sentence would exceed seq_length, save current and start new
            if current_length + sentence_length > self.seq_length and current_sequence:
                # Combine current sequence
                combined = torch.cat(current_sequence)
                padding_needed = self.seq_length - len(combined)
                
                if padding_needed > 0:
                    # Use cached padding tensor for efficiency
                    if padding_needed not in pad_tensor_cache:
                        pad_tensor_cache[padding_needed] = torch.full((padding_needed,), self.pad_token_id, dtype=torch.long)
                    combined = torch.cat([combined, pad_tensor_cache[padding_needed]])
                
                sequences.append(combined)
                current_sequence = []
                current_length = 0
                
                # Progress update
                if len(sequences) % 10000 == 0:
                    print(f"Created {len(sequences)} sequences so far...")
            
            # Add sentence to current sequence
            current_sequence.append(sentence)
            current_length += sentence_length
        
        # Add final sequence if it exists
        if current_sequence:
            combined = torch.cat(current_sequence)
            padding_needed = self.seq_length - len(combined)
            if padding_needed > 0:
                if padding_needed not in pad_tensor_cache:
                    pad_tensor_cache[padding_needed] = torch.full((padding_needed,), self.pad_token_id, dtype=torch.long)
                combined = torch.cat([combined, pad_tensor_cache[padding_needed]])
            sequences.append(combined)
        
        print(f"✅ Created {len(sequences)} multi-sentence sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a multi-sentence training example with MLM masking."""
        sequence = self.sequences[idx].clone()
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (sequence != self.pad_token_id).bool()
        
        # Apply MLM masking while respecting sentence boundaries
        masked_sequence, labels = self._apply_sentence_aware_masking(sequence)
        
        return {
            'input_ids': masked_sequence,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _apply_sentence_aware_masking(self, tokens):
        """Apply MLM masking while respecting sentence boundaries."""
        labels = torch.full_like(tokens, self.padding_label_id)
        
        # Find positions that can be masked (avoid special tokens and boundaries)
        maskable_positions = []
        for i, token_id in enumerate(tokens):
            # Skip special tokens
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            # Skip if this is a special token
            if token_id < self.n_special_tokens:
                continue
            # Don't mask tokens immediately adjacent to sentence boundaries
            if i > 0 and tokens[i-1] in [self.cls_token_id, self.sep_token_id]:
                continue
            if i < len(tokens) - 1 and tokens[i+1] in [self.cls_token_id, self.sep_token_id]:
                continue
                
            maskable_positions.append(i)
        
        if len(maskable_positions) == 0:
            return tokens, labels
        
        # Determine how many tokens to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_p))
        
        # Randomly select positions to mask
        import random
        masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
        for pos in masked_positions:
            labels[pos] = tokens[pos].clone()  # Store original token for loss
            
            rand = random.random()
            if rand < 0.8:
                # 80% of the time, replace with [MASK]
                tokens[pos] = self.mask_token_id
            elif rand < 0.9:
                # 10% of the time, replace with random token
                tokens[pos] = random.randint(self.n_special_tokens, self.tokenizer.get_vocab_size() - 1)
            # 10% of the time, keep unchanged
        
        return tokens, labels


if __name__ == "__main__":
    from tokenizers import Tokenizer

    # Test the original dataset
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    dataset = MlmDataset("data/pretrain/tokenized/train_0.pickle.gz", tokenizer, "whole_word")
    
    inputs, attention_mask, outputs = dataset[21000]

    for i, output in enumerate(outputs.tolist()):
        if output != -100:
            print(i, output, tokenizer.id_to_token(output))
    
    # Test the new sentence-aware dataset
    print("\n" + "="*50)
    print("Testing SentenceAwareDataset...")
    
    try:
        sentence_dataset = SentenceAwareDataset("./", tokenizer)
        if len(sentence_dataset) > 0:
            sample = sentence_dataset[0]
            print(f"Sample: {sample['input_ids'].shape}, {sample['attention_mask'].shape}, {sample['labels'].shape}")
            
            # Check for boundary tokens
            cls_positions = (sample['input_ids'] == tokenizer.token_to_id("[CLS]")).nonzero()
            sep_positions = (sample['input_ids'] == tokenizer.token_to_id("[SEP]")).nonzero()
            print(f"Boundary tokens: {len(cls_positions)} [CLS], {len(sep_positions)} [SEP]")
    except Exception as e:
        print(f"Could not test SentenceAwareDataset: {e}")


class BasicDataset(AbstractMlmDataset):
    def get_segment(self, index):
        document_index, segment_index = self.indexer.get_indices(index)
        tokens, _ = self.documents[document_index][segment_index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = self.seq_length - (len(tokens) + 2)
        padding = torch.full((padding_length,), fill_value=self.pad_index, dtype=torch.int16)
        segment = torch.cat([self.cls_index, tokens, self.sep_index, padding]).long()
        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])
        return segment, attention_mask, False



if __name__ == "__main__":
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    dataset = MlmDataset("data/pretrain/tokenized/train_0.pickle.gz", tokenizer, "whole_word")
    
    inputs, attention_mask, outputs = dataset[21000]

    for i, output in enumerate(outputs.tolist()):
        if output != -100:
            print(i, output, tokenizer.id_to_token(output))
