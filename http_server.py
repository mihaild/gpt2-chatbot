import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
import re
import logging
import argparse

import googletrans
import models
import regexps

logger = logging.getLogger(__name__)


class ServerState:
    def __init__(self, model, seconds_per_request: float):
        self.model = model
        self.seconds_per_request = seconds_per_request
        self.translator = googletrans.Translator()
        self.ok_query_re = re.compile(regexps.CORRECT_INPUT_RE)
        self.last = 0

    def time_to_wait(self):
        return self.seconds_per_request - (time.time() - self.last)

    def check_query(self, text):
        return self.ok_query_re.match(text)

    def update_last(self):
        self.time = time.time()


class RequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        state = self.server.state
        time_to_wait = state.time_to_wait()
        if time_to_wait > 0:
            logger.warning(f"Query early {time_to_wait:.2f}s")
            return self.handle_http(503, f"Please wait {time_to_wait:.2f} seconds")

        if len(self.path) > 1000:
            logger.warning(f"Request too long: {len(self.path)}")
            return self.handle_http(400, "Something is wrong, contact @mihaild")

        try:
            params = urlparse.parse_qs(urlparse.urlparse(self.path).query)
        except Exception:
            logger.warning("Couldn't parse http params")
            return self.handle_http(400, "Something is wrong, contact @mihaild")

        if 'source' in params and len(params['source']) == 1:
            logger.info(f"Request from {params['source'][0]}")

        if 'text' not in params:
            logger.warning("No text in params")
            return self.handle_http(400, "No text in params")

        if 'text' not in params:
            logger.warning("Something wrong with text in params")
            return self.handle_http(400, "No text in params")

        text = params['text'][0]
        if not state.check_query(text):
            logger.warning("Bad text")
            return self.handle_http(400, "Bad text: should be 3-150 symbols length and have only good characters")

        logger.info(f"original text: {text}")

        tr = state.translator.translate(text)
        translated = tr.text
        lang = tr.src

        logger.info(f"translated from: {lang}")
        logger.info(f"translation: {translated}")

        generated = state.model.generate(translated)
        n = '\n'
        logger.info(f"model response: {generated.replace(n, ' ')}")

        response = f"{translated} {generated}"

        if lang != "en":
            response_final = state.translator.translate(response, src="en", dest=lang).text
            logger.info(f"translated response: {response_final.replace(n, ' ')}")
        else:
            response_final = response

        return self.handle_http(200, response_final)

    def handle_http(self, status_code, text):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        content = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.wfile.write(bytes(content, 'UTF-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5901)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seconds-per-request", type=float, default=5.0)
    parser.add_argument("--model-dir", type=str, help="Path to model (empty to use dummy model)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    server_class = HTTPServer
    httpd = server_class((args.host, args.port), RequestHandler)
    if args.model_dir:
        model = models.GPT2Model(args.model_dir)
    else:
        model = models.DummyModel()

    httpd.state = ServerState(model, args.seconds_per_request)
    logger.info(f"Server starts on {args.host}:{args.port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

    logger.info("Server stops")


if __name__ == '__main__':
    main()
