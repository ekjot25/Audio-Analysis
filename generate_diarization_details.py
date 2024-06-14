import json
import re

def diarize_speakers_by_punctuation(transcribe_data):
    # Split transcript into sentences based on punctuation
    full_transcript = transcribe_data['results']['transcripts'][0]['transcript']
    sentences = re.split(r'(?<=[.?!])\s+', full_transcript)

    # Get all pronunciation items with their timestamps and speaker labels
    items = [item for item in transcribe_data['results']['items'] if item['type'] == 'pronunciation']
    words_with_time = [(item['alternatives'][0]['content'], float(item['start_time']), float(item['end_time']), item['speaker_label']) for item in items]

    diarized_output = []
    current_index = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_start_time = words_with_time[current_index][1] if current_index < len(words_with_time) else None
        sentence_end_time = None
        matched_words = []

        word_index = 0
        while word_index < len(sentence_words) and current_index < len(words_with_time):
            # Convert both to lower case to handle case insensitivity
            transcript_word = re.sub(r'[^a-zA-Z0-9]', '', sentence_words[word_index].lower())
            pronunciation_word = re.sub(r'[^a-zA-Z0-9]', '', words_with_time[current_index][0].lower())

            if pronunciation_word == transcript_word:
                matched_words.append(words_with_time[current_index])
                sentence_end_time = words_with_time[current_index][2]
                current_index += 1
            word_index += 1

        if matched_words:
            speaker_counts = {}
            for _, _, _, speaker in matched_words:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            if speaker_counts:
                dominant_speaker = max(speaker_counts, key=speaker_counts.get)
                if sentence_start_time and sentence_end_time:
                    diarized_output.append({
                        "speaker": dominant_speaker,
                        "text": sentence,
                        "start_time": sentence_start_time,
                        "end_time": sentence_end_time
                    })
        else:
            # If no words matched, log this sentence for review
            print(f"No matches found for sentence: {sentence}")

    return diarized_output
