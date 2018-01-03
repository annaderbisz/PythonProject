import unittest
from histogram import extract_color_histogram
import cv2

class histogramTest(unittest.TestCase):
    def testhistogram(self):
        img = cv2.imread("images\pies.jpg")
        histogram = extract_color_histogram(img)
        assert len(histogram) == 512, "Błąd histogramu"

if __name__ == "__main__":
    unittest.main()