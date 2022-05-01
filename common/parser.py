import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--usbcam', required=False)
argparser.add_argument('--videofile', required=False)
argparser.add_argument('-c', '--confidence', required=False)
argparser.add_argument('-f', '--font', required=False)
argparser.add_argument('--tsize', required=False)
argparser.add_argument('-d', '--device', required=False)

args = vars(argparser.parse_args())
