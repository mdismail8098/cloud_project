from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

chat_history = []

@csrf_exempt
def chat_view(request):
    response_text = ""
    if request.method == "POST":
        user_input = request.POST.get("user_input")

        memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        for msg in chat_history:
            memory.save_context({'input': msg['human']}, {'output': msg['AI']})

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful chatbot."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])

        groq_chat = ChatGroq(
            groq_api_key="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",  # Replace this!
            model_name="llama3-8b-8192"
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory
        )

        response_text = conversation.predict(human_input=user_input)
        chat_history.append({'human': user_input, 'AI': response_text})

    return render(request, "chat/index.html", {
        "chat_history": chat_history,
        "response": response_text
    })
