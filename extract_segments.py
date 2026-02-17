def extract_segments(states, frame_shift=0.01):

    segments = []
    T = len(states)

    in_speech = False
    start_frame = 0

    for t in range(T):

        if states[t] == 1 and not in_speech:
            # Start of speech
            in_speech = True
            start_frame = t

        elif states[t] == 0 and in_speech:
            # End of speech
            end_frame = t - 1
            start_time = start_frame * frame_shift
            end_time = (end_frame + 1) * frame_shift
            segments.append((start_time, end_time))
            in_speech = False

    # If file ends in speech
    if in_speech:
        end_frame = T - 1
        start_time = start_frame * frame_shift
        end_time = (end_frame + 1) * frame_shift
        segments.append((start_time, end_time))

    return segments

def post_process_segments(states,
                          frame_shift=0.01,
                          min_speech_duration=0.20,
                          min_silence_duration=0.10,
                          hangover=0.15):

    # Step 1: Extract raw segments
    raw_segments = []
    T = len(states)

    in_speech = False
    start_frame = 0

    for t in range(T):
        if states[t] == 1 and not in_speech:
            in_speech = True
            start_frame = t

        elif states[t] == 0 and in_speech:
            end_frame = t - 1
            raw_segments.append((start_frame, end_frame))
            in_speech = False

    if in_speech:
        raw_segments.append((start_frame, T - 1))

    # Convert frame to time
    segments = []
    for start_f, end_f in raw_segments:
        start_time = start_f * frame_shift
        end_time = (end_f + 1) * frame_shift
        segments.append([start_time, end_time])

    # Step 2: Remove short speech segments
    segments = [
        seg for seg in segments
        if (seg[1] - seg[0]) >= min_speech_duration
    ]

    if not segments:
        return []

    # Step 3: Merge close segments (short silence)
    merged = [segments[0]]

    for current in segments[1:]:
        prev = merged[-1]

        silence_gap = current[0] - prev[1]

        if silence_gap < min_silence_duration:
            # Merge
            prev[1] = current[1]
        else:
            merged.append(current)

    # Step 4: Apply hangover padding
    for seg in merged:
        seg[1] += hangover

    return merged
