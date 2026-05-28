"""Fix remaining broken emoji patterns in dashboard/app.py"""
import re

with open('dashboard/app.py', 'rb') as f:
    raw = f.read()

replacements = [
    # Captain flag: 🚩
    (b'flag = "\xef\xbf\xbd?\xef\xbf\xbd"', b'flag = "\xf0\x9f\x9a\xa9"'),
    # Analyse Transfers: 🔄
    (b'"\xef\xbf\xbd? Analyse Transfers"', b'"\xf0\x9f\x94\x84 Analyse Transfers"'),
    # Recommended Captain: 👑
    (b'Captain**: \xef\xbf\xbd?\xef\xbf\xbd', b'Captain**: \xf0\x9f\x91\x91'),
    # Preview Manager: 👁
    (b'"\xef\xbf\xbd? Preview Manager"', b'"\xf0\x9f\x91\x81 Preview Manager"'),
]

fixed = raw
for old, new in replacements:
    count = fixed.count(old)
    if count:
        print(f'  Replaced {count}x: {old[:30]}')
    fixed = fixed.replace(old, new)

# Report remaining
remaining = []
for m in re.finditer(b'\xef\xbf\xbd', fixed):
    line_start = fixed.rfind(b'\n', 0, m.start()) + 1
    line_end = fixed.find(b'\n', m.start())
    line = fixed[line_start:line_end]
    decoded = line.decode('utf-8', errors='replace').strip()
    if len(set(decoded.replace('?','').replace('#','').replace(' ',''))) > 3:
        if decoded not in remaining:
            remaining.append(decoded)

print(f'Remaining functional issues: {len(remaining)}')
for r in remaining:
    print(f'  {repr(r[:80])}')

with open('dashboard/app.py', 'wb') as f:
    f.write(fixed)
print('Saved.')
