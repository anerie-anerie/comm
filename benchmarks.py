"""
generate_benchmark.py - builds benchmark.jsonl: 100 synthetic prompts with
gold annotations. All data is fake (Faker + random), so no real PII is used.

Each line: {"id", "density", "text", "entities": [{"start","end","type","text"}]}
Run:  python3 generate_benchmark.py
"""
import json
import random
import re
import string

from faker import Faker

SEED = 192
random.seed(SEED)
Faker.seed(SEED)
fake = Faker("en_CA")

# real place names, because the NER model is trained on real text
LOCATIONS = [
    "Waterloo", "Toronto", "Kitchener", "Vancouver", "Montreal", "Ottawa", "Calgary",
    "Halifax", "Seattle", "Boston", "Chicago", "London", "Paris", "Berlin", "Mumbai",
    "Lagos", "Sao Paulo", "Karachi", "Sydney", "Tokyo", "Winnipeg", "Hamilton",
]


def gen_value(kind):
    if kind == "NAME":
        return f"{fake.first_name()} {fake.last_name()}"
    if kind == "LOCATION":
        return random.choice(LOCATIONS)
    if kind == "EMAIL":
        return fake.email()
    if kind == "IP":
        return fake.ipv4()
    if kind == "API_KEY":
        fmt = random.choice(["sk", "aws", "gh"])
        alnum = string.ascii_letters + string.digits
        if fmt == "sk":
            return "sk-" + "".join(random.choices(alnum, k=32))
        if fmt == "aws":
            return "AKIA" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        return "ghp_" + "".join(random.choices(alnum, k=36))
    raise ValueError(kind)


# {SLOT} markers are filled with fake values and recorded as gold entities.
PII_TEMPLATES = [
    # tutoring / school
    "Hi, I'm {NAME} and I live in {LOCATION}. Can you explain how pointers work in C?",
    "My professor {NAME} said my circuits lab was late. How do I ask for an extension politely?",
    # translation (Trust No Bot found PII is common here)
    "Translate to French: Dear {NAME}, please send the invoice to {EMAIL} before Friday.",
    "Translate this to Spanish: My name is {NAME} and I will arrive in {LOCATION} tomorrow.",
    # code editing / config
    "Fix this script:\nimport requests\nr = requests.get('http://{IP}/api', headers={{'Authorization': '{API_KEY}'}})\nprint(r.json())",
    "Why does my server at {IP} refuse connections? I set the key to {API_KEY} in .env",
    "Here is my config:\nHOST={IP}\nADMIN={EMAIL}\nTOKEN={API_KEY}\nCan you clean it up?",
    # emails / documents
    "Write a cover letter for {NAME}, who is applying in {LOCATION}. Contact: {EMAIL}.",
    "Draft an email to {NAME} ({EMAIL}) about moving our meeting from {LOCATION} to online.",
    "Summarize this note: {NAME} met {NAME} in {LOCATION} and shared the login {EMAIL}.",
    # dense
    "Help me debug. User {NAME} ({EMAIL}) in {LOCATION} reports the API at {IP} rejects key {API_KEY}.",
    "Resume review: {NAME}, {LOCATION}, {EMAIL}. Previously worked with {NAME} at a startup.",
]

# no PII, including hard negatives that look a little like PII
CLEAN_TEMPLATES = [
    "Explain the difference between a stack and a queue.",
    "How do I compute the Fourier transform of a square wave?",
    "What does the error 'segmentation fault' usually mean in C?",
    "Write a haiku about midterms.",
    "I upgraded to Python 3.11.4 and pip 23.1.2, but my install still fails.",   # version numbers
    "Set the firmware to version 1.2.3.4 and then reboot the board.",            # IP-like version
    "What is the time complexity of merge sort?",
    "Give me a study plan for linear algebra in two weeks.",
]

FILL = re.compile(r"\{([A-Z_]+)\}")


def build(template):
    # '{{' and '}}' are literal braces (used for Python dict syntax in templates)
    template = template.replace("{{", "\x00").replace("}}", "\x01")
    out, entities, cursor = [], [], 0
    pos = 0
    for m in FILL.finditer(template):
        literal = template[pos:m.start()]
        out.append(literal)
        cursor += len(literal.replace("\x00", "{").replace("\x01", "}"))
        value = gen_value(m.group(1))
        entities.append({"start": cursor, "end": cursor + len(value),
                         "type": m.group(1), "text": value})
        out.append(value)
        cursor += len(value)
        pos = m.end()
    out.append(template[pos:])
    text = "".join(out).replace("\x00", "{").replace("\x01", "}")
    return text, entities


def main(n=100, n_clean=15):
    rows = []
    for i in range(n):
        tpl = random.choice(CLEAN_TEMPLATES if i < n_clean else PII_TEMPLATES)
        text, ents = build(tpl)
        rows.append({"id": i, "density": len(ents), "text": text, "entities": ents})
    random.shuffle(rows)
    with open("benchmark.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(len(r["entities"]) for r in rows)
    print(f"Wrote {len(rows)} prompts with {total} gold entities to benchmark.jsonl")


if __name__ == "__main__":
    main()