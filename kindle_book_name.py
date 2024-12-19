import os
import mobi

book_fp = r'E:/kindle_content/Lenovo_R7000P2020H/My Kindle Content/B00IMSB8VM_EBOK/B00IMSB8VM_EBOK.azw'

def extract_metadata(file_path):
    try:
        book = mobi.Mobi(file_path)
        book.parse()
        metadata = book.get_metadata()
        print('have name')
        return metadata.get('title', 'Unknown Title')
    except Exception:
        print('no name')
        return None

if os.path.exists(book_fp) and os.path.isfile(book_fp) and book_fp.endswith('.azw'):
    print('have file')
else:
    print('no file')

if title := extract_metadata(book_fp):
    print(f'Title: {title}')
