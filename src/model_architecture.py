
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from src.config import LATENT_DIM, EMBEDDING_DIM

class Seq2SeqSpellChecker:
    """
    Seq2Seq LSTM Autoencoder Architecture.
    Trains correct-to-correct mapping with a tight bottleneck to learn French orthography.
    """
    def __init__(self, vocab_size, max_enc_len, max_dec_len):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        
        # Placeholders for structural layers
        self.trainer_model = None
        self.encoder_inputs = None
        self.encoder_states = None
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_embedding = None
        self.decoder_dense = None

    def build_trainer_model(self):
        """
        Builds the complete network for the Training teammate (uses Teacher Forcing).
        """
        # --- ENCODER ---
        self.encoder_inputs = Input(shape=(self.max_enc_len,), name="encoder_inputs")
        
        # Character Embedding Layer
        encoder_embed = Embedding(
            input_dim=self.vocab_size, 
            output_dim=EMBEDDING_DIM, 
            mask_zero=True, 
            name="encoder_emb"
        )
        
        # Dropout: Drops random character features during training.
        # This acts as an implicit "denoiser" to handle typos during inference.
        encoder_drop = Dropout(0.2, name="encoder_dropout")(encoder_embed(self.encoder_inputs))
        
        encoder_lstm = LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
        _, state_h, state_c = encoder_lstm(encoder_drop)
        self.encoder_states = [state_h, state_c]

        # --- DECODER ---
        self.decoder_inputs = Input(shape=(self.max_dec_len,), name="decoder_inputs")
        
        self.decoder_embedding = Embedding(
            input_dim=self.vocab_size, 
            output_dim=EMBEDDING_DIM, 
            mask_zero=True, 
            name="decoder_emb"
        )
        decoder_embedded = self.decoder_embedding(self.decoder_inputs)
        
        # The Decoder LSTM state is initialized by the Encoder's final state bottleneck
        self.decoder_lstm = LSTM(
            LATENT_DIM, 
            return_sequences=True, 
            return_state=True, 
            name="decoder_lstm"
        )
        decoder_outputs, _, _ = self.decoder_lstm(decoder_embedded, initial_state=self.encoder_states)
        
        # Dense Softmax layer projects output to the size of the alphabet
        self.decoder_dense = Dense(self.vocab_size, activation='softmax', name="decoder_dense")
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # --- WRAP INTO TRAINER MODEL ---
        self.trainer_model = Model(
            inputs=[self.encoder_inputs, self.decoder_inputs], 
            outputs=decoder_outputs, 
            name="seq2seq_trainer"
        )
        return self.trainer_model

    def extract_inference_models(self):
        """
        Extracts standalone, step-by-step components for the Inference teammate.
        Crucial: Must only be called after trainer_model has been trained/compiled.
        """
        if self.trainer_model is None:
            raise ValueError("The training model must be built before extracting inference hooks.")

        # 1. Standalone Encoder Hook (Outputs context states from misspelled word)
        encoder_engine = Model(
            inputs=self.encoder_inputs, 
            outputs=self.encoder_states, 
            name="inference_encoder"
        )

        # 2. Standalone Decoder Hook (Decodes step-by-step in a loop)
        latent_state_h = Input(shape=(LATENT_DIM,), name="input_state_h")
        latent_state_c = Input(shape=(LATENT_DIM,), name="input_state_c")
        decoder_states_inputs = [latent_state_h, latent_state_c]

        # Use the already trained embedding layer weights
        dec_emb_out = self.decoder_embedding(self.decoder_inputs)
        
        # Use the already trained LSTM layers, tracking updated internal states
        dec_lstm_out, state_h_dec, state_c_dec = self.decoder_lstm(
            dec_emb_out, 
            initial_state=decoder_states_inputs
        )
        decoder_states_outputs = [state_h_dec, state_c_dec]
        
        # Use the already trained Dense layer to output token probabilities
        dec_dense_out = self.decoder_dense(dec_lstm_out)

        decoder_engine = Model(
            inputs=[self.decoder_inputs] + decoder_states_inputs,
            outputs=[dec_dense_out] + decoder_states_outputs,
            name="inference_decoder"
        )

        return encoder_engine, decoder_engine