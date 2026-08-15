MORSE_CODE_DICT = {
    ".-": "ا",
    "-...": "ب",
    "-": "ت",
    "-.-.": "ث",
    ".---": "ج",
    "....": "ح",
    "---": "خ",
    "-..": "د",
    "--..": "ذ",
    ".-.": "ر",
    "---.": "ز",
    "...": "س",
    "----": "ش",
    "-..-": "ص",
    "...-": "ض",
    "..-": "ط",
    "-.--": "ظ",
    ".-.-": "ع",
    "--.": "غ",
    "..-.": "ف",
    "--.-": "ق",
    "-.-": "ك",
    ".-..": "ل",
    "--": "م",
    "-.": "ن",
    "..-..": "ه",
    ".--": "و",
    "..": "ي",
}


def decode_morse(morse_str):

    if not morse_str:
        return ""

    words = morse_str.strip().split(" / ")

    decoded_message = []

    for word in words:

        letters = word.split(" ")

        decoded_word = "".join(
            MORSE_CODE_DICT.get(
                letter,
                ""
            )
            for letter in letters
        )

        decoded_message.append(
            decoded_word
        )

    return " ".join(
        decoded_message
    )