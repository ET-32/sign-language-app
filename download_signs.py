"""
Run once: downloads ASL sign GIFs from Lifeprint into assets/signs/
Usage: python download_signs.py
"""
import os, time, urllib.request as req

OUT = os.path.join(os.path.dirname(__file__), 'assets', 'signs')
os.makedirs(OUT, exist_ok=True)

HEADERS = {
    'Referer': 'https://www.lifeprint.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
}

def fetch(url, path):
    try:
        r = req.Request(url, headers=HEADERS)
        with req.urlopen(r, timeout=10) as resp:
            data = resp.read()
            if len(data) < 200:
                return False
            with open(path, 'wb') as f:
                f.write(data)
        return True
    except Exception as e:
        print(f'  error: {e}')
        return False

# Common words
WORDS = [
    'hello', 'goodbye', 'thankyou', 'please', 'sorry',
    'yes', 'no', 'help', 'stop', 'water', 'good', 'bad',
    'more', 'you', 'love', 'where', 'what', 'name',
    'eat', 'drink', 'family', 'friend', 'home', 'i',
]

print('Downloading common signs...')
for word in WORDS:
    path = os.path.join(OUT, f'{word}.gif')
    if os.path.exists(path):
        print(f'  skip {word} (exists)')
        continue
    url = f'http://www.lifeprint.com/asl101/gifs-animated/{word}.gif'
    ok = fetch(url, path)
    print(f'  {"OK" if ok else "FAIL"} {word}')
    time.sleep(0.4)

# Alphabet
print('\nDownloading alphabet...')
for ch in 'abcdefghijklmnopqrstuvwxyz':
    path = os.path.join(OUT, f'{ch}.gif')
    if os.path.exists(path):
        print(f'  skip {ch} (exists)')
        continue
    url = f'http://www.lifeprint.com/asl101/fingerspelling/abc-gifs/{ch}.gif'
    ok = fetch(url, path)
    print(f'  {"OK" if ok else "FAIL"} {ch}')
    time.sleep(0.3)

print('\nDone. Files in:', OUT)
files = [f for f in os.listdir(OUT) if f.endswith('.gif')]
print(f'{len(files)} GIFs downloaded.')
