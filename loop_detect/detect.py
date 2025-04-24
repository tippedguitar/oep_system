from PIL import Image
import threading


class VideoLoopFinder:

    def __init__(self, RES=32):
        self.RES = 32
        self.OPT_VALUE = 1
        self.MAX_LIMIT = 1000
        self.THRESHOLD = 10 // self.OPT_VALUE
        self.seen_frames = {}
        self.duplicate_frames = {}
        self.current_frames = []

        self.finish = True
        self.lock = threading.Lock()

    def ahash(self, frame, res=64):
        i = Image.fromarray(frame)
        i = i.resize((res, res), Image.ANTIALIAS).convert('L')
        pixels = list(i.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pixels))
        hexadecimal = int(bits, 2).__format__('016x').upper()
        return hexadecimal

    def find_duplicates(self, vid, res=32):

        all_frames = len(vid)
        print(all_frames)

        for x in range(0, all_frames // self.OPT_VALUE, self.OPT_VALUE):

            frame = vid[x]

            hashed = self.ahash(frame, res)

            if self.seen_frames.get(hashed, None):

                self.duplicate_frames[hashed].append(x)
            else:

                self.seen_frames[hashed] = x
                self.duplicate_frames[hashed] = [x]
        # for i in self.duplicate_frames:
        # 	print(i)
        duplicates = [abs(self.duplicate_frames[x][0] - self.duplicate_frames[x][-1]) for x in self.duplicate_frames if
                      len(self.duplicate_frames[x]) > 1]
        print(duplicates)
        return duplicates

    def get_valid_duplicates(self, duplicate_frames):
        s = 0
        loop_frame_count = 0
        for x in duplicate_frames:
            if x > self.THRESHOLD:
                loop_frame_count += 1

        print("duplicate frame count ->  ", loop_frame_count)

        if len(self.seen_frames) > self.MAX_LIMIT:
            self.seen_frames.clear()
            self.duplicate_frames.clear()
        if loop_frame_count > self.THRESHOLD:
            print("LOOP DETECTED")
            self.seen_frames.clear()
            self.duplicate_frames.clear()
            return [True, loop_frame_count]
        else:
            print("NO LOOP")
        return [False, loop_frame_count]


print("detect imported")