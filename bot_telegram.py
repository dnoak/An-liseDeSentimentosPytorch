from PIL import Image,ImageStat
import test_model
import os
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import sys

MEU_TOKEN= '5439938561:AAHg519kCsBG2YPbixeYK9Zc21TqHFPFj64'

# Pasta para imagens enviadas pelo usuário
pasta_imgs='./Telegram_Imagens_Recebidas/' 

print('Carregando BOT usando o token ',MEU_TOKEN)


def echo(update, context):
    
    mensagem=test_model.response(update.message.text)
    
    if mensagem == 1:
        resposta = f'positivo'
    else:
        resposta = f'negativo'

    update.message.reply_text(f'Comentário {resposta}')

def start(update, context):
    msg = '''
    Olá, digite algum comentário para eu analisar, ou
    se quiser, pegue algum review de filme e coloque
    aqui que eu analisarei:
    https://www.adorocinema.com
    '''
    update.message.reply_text()


def main():

    # Cria o módulo que vai ficar lendo o que está sendo escrito
    # no seu chatbot e respondendo.
    # Troque TOKEN pelo token que o @botfather te passou (se
    # ainda não usou @botfather, leia novamente o README)
    updater = Updater(MEU_TOKEN, use_context=True)

    # Cria o submódulo que vai tratar cada mensagem recebida
    dp = updater.dispatcher

    # Define as funções que vão ser ativadas com /start e /help
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # Define a função que vai tratar os textos
    dp.add_handler(MessageHandler(Filters.text, echo))

    # Cria pasta para as imagens enviadas pelo usuário

    # Inicia o chatbot
    updater.start_polling()

    # Roda o bot até que você dê um CTRL+C
    updater.idle()


if __name__ == '__main__':
    print('Bot respondendo, use CRTL+C para parar')
    main()

