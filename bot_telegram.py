import requests
import re
import logging
import argparse
import sys

from telegram.ext import Updater, Filters
import telegram.ext
import telegram

import models
import regexps

logger = logging.getLogger(__name__)

OK_RE = re.compile(f'^(?:клиппи|клипи|clippy) ({regexps.CORRECT_INPUT_RE})', re.I)


class HttpModel:
    def __init__(self, host):
        self.host = host

    def generate(self, text):
        logger.info(f"Sending request {text} to {self.host}")
        resp = requests.get(self.host, {"text": text, "source": "telegram"})
        resp_text = resp.text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
        return resp_text


class ClippyHandler:
    def __init__(self, model, ok_chats):
        self.model = model
        self.ok_chats = ok_chats

    def __call__(self, bot, update):
        if self.ok_chats and update.message.chat.id not in self.ok_chats:
            logger.info(f"Message {update.message.text} in bad chat {update.message.chat.id}")
            bot.send_message(chat_id=update.message.chat_id, text="Меня здесь быть не должно.")
            if update.message.chat.type != 'private':
                bot.leave_chat(chat_id=update.message.chat_id)
            return
        m = OK_RE.match(update.message.text)
        if m is None:
            return
        logger.info(f"Text {m.group(1)} in chat {update.message.chat.id}")
        text = m.group(1)
        bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
        resp_text = self.model.generate(text)
        n = '\n'
        logger.info(f"Response {resp_text.replace(n, ' ')}")
        bot.send_message(chat_id=update.message.chat_id, text=resp_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2-host", type=str, help="host:port to use model via http requests")
    parser.add_argument("--model-dir", type=str, help="directory with gpt2 model to load it directly in bot")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--bot-token", type=str, required=True)
    parser.add_argument("--ok-chats", metavar="CHAT_ID", type=int, nargs='+', help="List of chats to response in (empty to response everywhere")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if (args.gpt2_host is not None) and (args.model_dir is not None):
        sys.stderr.write("At most one of gpt2-host and model-dir should be specified\n")
        exit(1)

    if args.gpt2_host is not None:
        model = HttpModel(args.gpt2_host)
    elif args.model_dir is not None:
        model = models.GPT2Model(args.model_dir)
    else:
        model = models.DummyModel()

    updater = Updater(args.bot_token)
    handler = ClippyHandler(model, args.ok_chats)
    dp = updater.dispatcher
    dp.add_handler(telegram.ext.MessageHandler(Filters.text, handler))
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
