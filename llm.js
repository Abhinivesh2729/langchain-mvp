import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from "dotenv"
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"; 
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

dotenv.config()

//creating model
const model = new ChatGoogleGenerativeAI({
    //model: "Gemini 1.5 Flash-8B",
    temperature: 0.1
})

//creating prompt
const prompt = ChatPromptTemplate.fromTemplate(
    "Aser the user's question {input} "
)

//output parser
const parser = new StringOutputParser()

//basic chain

const chian = prompt.pipe(model).pipe(parser)


//create doc loader
const loader = new CheerioWebBaseLoader("https://in.linkedin.com/company/atmega-erode")
const docs = await loader.load()

//text splitter
const spilitter = new RecursiveCharacterTextSplitter({chunkSize: 100, chunkOverlap: 20})
const splitDocs = await spilitter.splitDocuments(docs)

//create embeddings
const embeddings = new GoogleGenerativeAIEmbeddings()

//create vector store db
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings)

//create retriver
const retriver = vectorStore.asRetriever({k:2})


//create retrival chain
const retrivalChain = await createRetrievalChain({
    combineDocsChain: chian,
    retriever: retriver
})

// output
const resposne = await retrivalChain.invoke({'input': "do you know of ATmega Software Technologies LLP"})
console.log(resposne)