"""
music_generator.py
==================
AI Music Composition Engine for B.Tech Project:
"Development of an AI-Based Music Composition System Using Machine Learning Techniques"

This module loads the trained LSTM Keras model and the preprocessing metadata,
validates input shapes, generates new music note-by-note (autoregression),
and writes the output to a standard MIDI file using `pretty_midi`.
"""

import os
import json
import numpy as np
import keras
import pretty_midi


class MusicGenerator:
    """
    Music Generator Engine using a trained Keras multi-output LSTM model.
    """

    def __init__(
        self,
        model_path="models/ai_music_model.keras",
        preprocessing_path="models/music_preprocessing.json",
    ):
        """
        Initializes the generator by loading configuration and the trained model.
        """
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path

        # 1. Load preprocessing settings
        self._load_preprocessing()

        # 2. Load the trained Keras model and validate shapes
        self._load_and_validate_model()

    def _load_preprocessing(self):
        """
        Loads the preprocessing JSON file and sets up pitch mappings and scales.
        """
        if not os.path.exists(self.preprocessing_path):
            raise FileNotFoundError(
                f"Preprocessing configuration file not found at: {self.preprocessing_path}"
            )

        with open(self.preprocessing_path, "r", encoding="utf-8") as f:
            self.preprocessing_config = json.load(f)

        # Extract settings
        self.unique_pitches = self.preprocessing_config.get("unique_pitches", [])
        if not self.unique_pitches:
            raise ValueError("unique_pitches list is missing or empty in preprocessing JSON.")

        self.num_pitches = len(self.unique_pitches)
        self.duration_scale = float(self.preprocessing_config.get("duration_scale", 1.5))
        self.gap_scale = float(self.preprocessing_config.get("gap_scale", 0.5))
        self.sequence_length = int(self.preprocessing_config.get("sequence_length", 20))
        self.num_features = int(self.preprocessing_config.get("num_features", 4))

        # Build bidirectional mappings:
        # pitch_to_id: MIDI pitch (e.g., 60) -> discrete class ID [0 .. 70]
        # id_to_pitch: discrete class ID [0 .. 70] -> MIDI pitch (e.g., 60)
        self.pitch_to_id = {pitch: idx for idx, pitch in enumerate(self.unique_pitches)}
        self.id_to_pitch = {idx: pitch for idx, pitch in enumerate(self.unique_pitches)}

        print(f"[MusicGenerator] Preprocessing loaded:")
        print(f"  - Unique pitches: {self.num_pitches} (range: {min(self.unique_pitches)} to {max(self.unique_pitches)})")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Duration scale: {self.duration_scale}")
        print(f"  - Gap scale: {self.gap_scale}")
        print(f"  - Number of features: {self.num_features}")

    def _load_and_validate_model(self):
        """
        Loads the trained Keras model and validates input/output layer dimensions.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        print(f"[MusicGenerator] Loading trained model from '{self.model_path}'...")
        self.model = keras.models.load_model(self.model_path)

        # Validate Input Shape: expected (None, sequence_length, num_features)
        input_shape = self.model.input_shape
        expected_input_shape = (None, self.sequence_length, self.num_features)
        if input_shape != expected_input_shape:
            raise ValueError(
                f"Model input shape mismatch! Expected {expected_input_shape}, but got {input_shape}."
            )

        # Validate Output Shapes:
        # Output 0 (pitch): (None, num_pitches)
        # Output 1 (duration): (None, 1)
        # Output 2 (velocity): (None, 1)
        # Output 3 (gap): (None, 1)
        output_shapes = self.model.output_shape
        if not isinstance(output_shapes, list) or len(output_shapes) != 4:
            raise ValueError(
                f"Model must have 4 output heads [pitch, duration, velocity, gap]. Got: {output_shapes}"
            )

        expected_pitch_shape = (None, self.num_pitches)
        if output_shapes[0] != expected_pitch_shape:
            raise ValueError(
                f"Pitch output shape mismatch! Expected {expected_pitch_shape}, got {output_shapes[0]}."
            )

        for i, name in enumerate(["duration", "velocity", "gap"], start=1):
            if output_shapes[i] != (None, 1):
                raise ValueError(
                    f"{name.capitalize()} output shape mismatch! Expected (None, 1), got {output_shapes[i]}."
                )

        print("[MusicGenerator] Model loaded successfully. Input and output shapes verified:")
        print(f"  - Input: {input_shape}")
        print(f"  - Outputs: pitch={output_shapes[0]}, duration={output_shapes[1]}, velocity={output_shapes[2]}, gap={output_shapes[3]}")

    def validate_seed(self, seed_array):
        """
        Validates that a seed sequence has the required shape and representation.

        Requirements:
        - Shape: (20, 4)
        - Column 0 (pitch_id): Float/Int in range [0, num_pitches - 1]
        - Column 1 (normalized_duration): Non-negative float (typically 0.0 to 1.0)
        - Column 2 (normalized_velocity): Float in range [0.0, 1.0] (velocity / 127)
        - Column 3 (normalized_gap): Non-negative float (typically 0.0 to 1.0)
        """
        if not isinstance(seed_array, np.ndarray):
            seed_array = np.array(seed_array, dtype=np.float32)

        if seed_array.shape != (self.sequence_length, self.num_features):
            raise ValueError(
                f"Seed shape must be ({self.sequence_length}, {self.num_features}), got {seed_array.shape}."
            )

        # Verify pitch_id bounds
        pitch_ids = seed_array[:, 0]
        if np.any(pitch_ids < 0) or np.any(pitch_ids >= self.num_pitches):
            raise ValueError(
                f"Seed contains pitch IDs outside valid range [0, {self.num_pitches - 1}]."
            )

        # Verify velocity bounds
        velocities = seed_array[:, 2]
        if np.any(velocities < 0.0) or np.any(velocities > 1.0):
            raise ValueError(
                "Seed contains normalized velocity values outside [0.0, 1.0]. "
                "Ensure velocity is normalized by dividing raw velocity by 127.0."
            )

        return seed_array.astype(np.float32)

    def sample_pitch(self, probabilities, temperature=1.0):
        """
        Samples a pitch-class ID from predicted probability distribution using temperature.

        - temperature <= 0: Greedy selection (argmax) for deterministic output.
        - temperature = 1.0: Standard model probability distribution.
        - temperature < 1.0: More conservative, higher-probability choices.
        - temperature > 1.0: More diverse, creative, exploratory choices.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if temperature <= 0.05:
            # Deterministic: pick highest probability
            return int(np.argmax(probabilities))

        # Apply temperature scaling to logits
        # Adding a small epsilon prevents log(0)
        epsilon = 1e-8
        log_probs = np.log(np.clip(probabilities, epsilon, 1.0))
        scaled_logits = log_probs / float(temperature)

        # Softmax re-normalization
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        scaled_probs = exp_logits / np.sum(exp_logits)

        # Sample from the adjusted distribution
        pitch_id = np.random.choice(len(scaled_probs), p=scaled_probs)
        return int(pitch_id)

    def generate_notes(self, seed_sequence, num_notes=50, temperature=1.0):
        """
        Generates a sequence of musical notes using the autoregressive generation loop.

        Parameters:
        - seed_sequence: Array of shape (20, 4) in the training representation:
          [pitch_id, normalized_duration, normalized_velocity, normalized_gap]
        - num_notes: Number of new notes to generate (int > 0).
        - temperature: Pitch-sampling temperature (float > 0).

        Returns:
        - List of note dictionaries:
          [{'pitch': int, 'duration': float, 'velocity': int, 'gap': float}, ...]
        """
        if num_notes <= 0:
            raise ValueError("num_notes must be a positive integer.")

        # Validate the seed sequence
        validated_seed = self.validate_seed(seed_sequence)

        # Current working window of shape (1, 20, 4)
        current_sequence = np.copy(validated_seed)

        generated_notes = []

        print(f"[MusicGenerator] Starting generation of {num_notes} notes (temperature={temperature:.2f})...")

        for step in range(num_notes):
            # Model expects input of shape (1, 20, 4)
            model_input = np.expand_dims(current_sequence, axis=0)

            # Predict next note properties
            preds = self.model.predict(model_input, verbose=0)

            # Extract outputs from the 4 heads
            # Head 0: Pitch probabilities of shape (1, 71)
            pitch_probs = preds[0][0]
            # Head 1: Normalized duration of shape (1, 1) -> Sigmoid [0..1]
            raw_duration = float(preds[1][0][0])
            # Head 2: Normalized velocity of shape (1, 1) -> Sigmoid [0..1]
            raw_velocity = float(preds[2][0][0])
            # Head 3: Normalized gap of shape (1, 1) -> Sigmoid [0..1]
            raw_gap = float(preds[3][0][0])

            # 1. Sample next pitch ID and map back to real MIDI pitch
            next_pitch_id = self.sample_pitch(pitch_probs, temperature=temperature)
            real_midi_pitch = self.id_to_pitch[next_pitch_id]

            # 2. Denormalize duration (duration_scale = 1.5s, minimum threshold 0.05s)
            actual_duration = max(0.05, raw_duration * self.duration_scale)

            # 3. Denormalize velocity (scale by 127, clip to valid MIDI range [1, 127])
            actual_velocity = int(np.clip(round(raw_velocity * 127.0), 1, 127))

            # 4. Denormalize gap (gap_scale = 0.5s, minimum threshold 0.01s)
            actual_gap = max(0.01, raw_gap * self.gap_scale)

            # Save generated note
            note_data = {
                "pitch": real_midi_pitch,
                "pitch_id": next_pitch_id,
                "duration": round(actual_duration, 4),
                "velocity": actual_velocity,
                "gap": round(actual_gap, 4),
            }
            generated_notes.append(note_data)

            # Prepare the new note feature vector for the input representation:
            # [pitch_id, normalized_duration, normalized_velocity, normalized_gap]
            # We feed the model its own normalized predictions for duration, velocity, gap
            next_input_vector = np.array(
                [float(next_pitch_id), raw_duration, raw_velocity, raw_gap],
                dtype=np.float32,
            )

            # Slide window: drop oldest note (index 0) and append the new note
            current_sequence = np.vstack([current_sequence[1:], next_input_vector])

        print(f"[MusicGenerator] Successfully generated {len(generated_notes)} notes.")
        return generated_notes

    def save_to_midi(
        self,
        notes,
        output_file="generated_music.mid",
        instrument_program=0,
        bpm=120.0,
    ):
        """
        Converts the list of generated note dictionaries into a MIDI file using `pretty_midi`.

        Parameters:
        - notes: List of note dictionaries containing 'pitch', 'duration', 'velocity', 'gap'.
        - output_file: Destination .mid file path.
        - instrument_program: MIDI program number (0 = Acoustic Grand Piano).
        - bpm: Initial tempo in beats per minute.
        """
        if not notes:
            raise ValueError("No notes to save.")

        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

        # Create an Instrument instance (program 0 = Acoustic Grand Piano)
        instrument = pretty_midi.Instrument(
            program=instrument_program,
            is_drum=False,
            name="AI Piano",
        )

        current_time = 0.0

        for note_dict in notes:
            pitch = int(note_dict["pitch"])
            duration = float(note_dict["duration"])
            velocity = int(note_dict["velocity"])
            gap = float(note_dict["gap"])

            start_time = current_time
            end_time = start_time + duration

            # Ensure note end is strictly after start
            if end_time <= start_time:
                end_time = start_time + 0.05

            midi_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(midi_note)

            # In the training data, 'gap' is the onset-to-onset time
            # (time from current note start to next note start)
            current_time += gap

        # Add instrument to PrettyMIDI
        midi.instruments.append(instrument)

        # Ensure directory exists
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write to MIDI file
        midi.write(output_file)
        print(f"[MusicGenerator] MIDI saved successfully to '{output_file}'.")
        return output_file

    def convert_midi_to_seed(self, midi_path, num_notes=20):
        """
        Helper method to extract the first 20 notes from a MIDI file and convert them
        strictly into the exact training representation:
        [pitch_id, duration / duration_scale, velocity / 127.0, gap / gap_scale]

        Note: multi_song_generated_music.mid is a generated sample, but its notes
        can be transformed into a valid seed via this method.
        """
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found at: {midi_path}")

        midi = pretty_midi.PrettyMIDI(midi_path)
        if not midi.instruments:
            raise ValueError(f"No instruments found in MIDI file: {midi_path}")

        # Collect notes from instruments
        all_notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)

        if not all_notes:
            all_notes = midi.instruments[0].notes

        # Sort chronologically by start time
        all_notes.sort(key=lambda n: n.start)

        if len(all_notes) < num_notes:
            raise ValueError(
                f"MIDI file contains only {len(all_notes)} notes, but {num_notes} are required for seed."
            )

        seed_rows = []
        for i in range(num_notes):
            n = all_notes[i]
            # Duration in seconds
            raw_dur = max(0.01, n.end - n.start)
            # Gap (onset-to-onset delta)
            if i < num_notes - 1:
                raw_gap = max(0.01, all_notes[i + 1].start - n.start)
            else:
                raw_gap = 0.1  # Default for the last note in sequence

            # Pitch mapping
            if n.pitch in self.pitch_to_id:
                pitch_id = self.pitch_to_id[n.pitch]
            else:
                # Find nearest pitch in unique_pitches vocabulary
                nearest_pitch = min(self.unique_pitches, key=lambda p: abs(p - n.pitch))
                pitch_id = self.pitch_to_id[nearest_pitch]

            # Normalize features exactly according to training pipeline
            norm_pitch_id = float(pitch_id)
            norm_dur = float(np.clip(raw_dur / self.duration_scale, 0.0, 1.0))
            norm_vel = float(np.clip(n.velocity / 127.0, 0.0, 1.0))
            norm_gap = float(np.clip(raw_gap / self.gap_scale, 0.0, 1.0))

            seed_rows.append([norm_pitch_id, norm_dur, norm_vel, norm_gap])

        return np.array(seed_rows, dtype=np.float32)


# =====================================================================
# Standalone Helper: Construct a Valid Seed from Raw Musical Notes
# =====================================================================

def create_valid_seed_from_notes(generator, raw_notes):
    """
    Constructs a verified seed array of shape (20, 4) in the original training
    representation from a list of 20 raw musical note definitions.

    Parameters:
    - generator: Initialized MusicGenerator instance.
    - raw_notes: List of 20 tuples/dicts:
      [(pitch, duration_seconds, velocity, gap_seconds), ...]

    Returns:
    - np.ndarray of shape (20, 4) with dtype float32.
    """
    if len(raw_notes) != generator.sequence_length:
        raise ValueError(
            f"Expected exactly {generator.sequence_length} notes for seed, got {len(raw_notes)}."
        )

    seed_matrix = []
    for item in raw_notes:
        if isinstance(item, dict):
            pitch = item["pitch"]
            duration = item["duration"]
            velocity = item["velocity"]
            gap = item["gap"]
        else:
            pitch, duration, velocity, gap = item

        # Check pitch vocabulary
        if pitch not in generator.pitch_to_id:
            valid_list_preview = generator.unique_pitches[:10]
            raise ValueError(
                f"Pitch {pitch} is not in unique_pitches! (Valid examples: {valid_list_preview}...)"
            )

        pitch_id = float(generator.pitch_to_id[pitch])
        norm_duration = float(np.clip(duration / generator.duration_scale, 0.0, 1.0))
        norm_velocity = float(np.clip(velocity / 127.0, 0.0, 1.0))
        norm_gap = float(np.clip(gap / generator.gap_scale, 0.0, 1.0))

        seed_matrix.append([pitch_id, norm_duration, norm_velocity, norm_gap])

    seed_array = np.array(seed_matrix, dtype=np.float32)
    return generator.validate_seed(seed_array)


def get_default_musical_seed(generator):
    """
    Provides a musically coherent, 20-note classical melody seed
    (C Major / A Minor arpeggios and motifs) converted into the exact
    training representation.

    All 20 pitches exist in generator.unique_pitches.
    """
    # 20 notes: (MIDI pitch, duration in seconds, velocity [0..127], gap in seconds)
    # Using notes from unique_pitches: 60(C4), 62(D4), 64(E4), 65(F4), 67(G4), 69(A4), 71(B4), 72(C5)
    default_melody = [
        (60, 0.35, 100, 0.12),  # C4
        (64, 0.35, 102, 0.12),  # E4
        (67, 0.35, 104, 0.12),  # G4
        (72, 0.40, 105, 0.15),  # C5
        (71, 0.30, 102, 0.12),  # B4
        (69, 0.30, 100, 0.12),  # A4
        (67, 0.35, 102, 0.14),  # G4
        (65, 0.30, 98,  0.12),  # F4
        (64, 0.35, 100, 0.12),  # E4
        (62, 0.30, 98,  0.12),  # D4
        (60, 0.40, 102, 0.16),  # C4
        (64, 0.30, 100, 0.12),  # E4
        (67, 0.35, 102, 0.12),  # G4
        (69, 0.35, 104, 0.14),  # A4
        (72, 0.45, 106, 0.18),  # C5
        (74, 0.35, 104, 0.14),  # D5
        (72, 0.35, 102, 0.12),  # C5
        (69, 0.30, 100, 0.12),  # A4
        (67, 0.35, 102, 0.14),  # G4
        (64, 0.40, 100, 0.15),  # E4
    ]

    return create_valid_seed_from_notes(generator, default_melody)


# =====================================================================
# Main Block for Standalone Testing
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("AI Music Composition System - Music Generator Engine Test")
    print("=" * 65)

    # 1. Initialize Generator
    gen = MusicGenerator()

    # 2. Get a valid seed sequence (in training representation)
    seed = get_default_musical_seed(gen)
    print(f"\n[Test] Seed created successfully. Shape: {seed.shape}")
    print("[Test] First 3 notes of seed representation:")
    for i in range(3):
        print(f"  Note {i}: pitch_id={seed[i, 0]:.0f} (MIDI {gen.id_to_pitch[int(seed[i, 0])]}), "
              f"norm_dur={seed[i, 1]:.3f}, norm_vel={seed[i, 2]:.3f}, norm_gap={seed[i, 3]:.3f}")

    # 3. Generate 30 notes with temperature = 0.8
    num_test_notes = 30
    temp = 0.8
    print(f"\n[Test] Generating {num_test_notes} notes with temperature {temp}...")
    notes = gen.generate_notes(seed, num_notes=num_test_notes, temperature=temp)

    # 4. Save to test MIDI file
    output_path = "test_generated_music.mid"
    gen.save_to_midi(notes, output_file=output_path)
    print(f"\n[Test] Success! Generated MIDI file created at: {output_path}")
    print("=" * 65)
