import torch.nn as nn


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        assert isinstance(encoder, Seq2SeqEncoder)
        assert isinstance(decoder, Seq2SeqDecoder)
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Build a new model instance."""
        raise NotImplementedError

    def forward(self, src_tokens, src_lengths, tgt_inputs):

        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(tgt_inputs, encoder_out)
        return decoder_out

class MMSeq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        assert isinstance(encoder, Seq2SeqEncoder)
        assert isinstance(decoder, Seq2SeqDecoder)
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Build a new model instance."""
        raise NotImplementedError

    def forward(self, src_tokens, src_lengths, tgt_inputs, video_inputs):

        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(tgt_inputs, encoder_out, video_inputs)
        return decoder_out

class Seq2SeqEncoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths):
        raise NotImplementedError


class Seq2SeqDecoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, tgt_inputs, encoder_out):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state. This should be called when the order of the input has changed from the previous
        time step. A typical use case is beam search, where the input order changes between time steps based on the
        selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(incremental_state, new_order)
        self.apply(apply_reorder_incremental_state)

class Source2TargetModel(nn.Module):
    def __init__(self, source_encoder, target_encoder):
        super().__init__()
        assert isinstance(source_encoder, Seq2SeqEncoder)
        assert isinstance(target_encoder, Seq2SeqEncoder)
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Build a new model instance."""
        raise NotImplementedError

    def forward(self, src_en_tokens, src_en_lengths, src_pt_tokens, src_pt_lengths):
        
        source_encoder_out = self.source_encoder(src_pt_tokens, src_pt_lengths)
        target_encoder_out = self.target_encoder(source_encoder_out, src_en_tokens, src_en_lengths)

        return target_encoder_out
