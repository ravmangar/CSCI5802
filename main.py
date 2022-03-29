from get_waveforms import MINISEED_Downloader

def main():
    files = MINISEED_Downloader()
    files.download()

if __name__ == "__main__":
    main()