import os
import shutil
import urllib.request
# pyrefly: ignore [missing-import]
import chromadb
import streamlit as st
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# =========================================================
# 1. STREAMLIT CONFIGURATION & PERSISTENT INFRASTRUCTURE
# =========================================================
st.set_page_config(
    page_title="Robotics Copilot RAG v2",
    page_icon="🤖",
    layout="wide"
)

DIRECTORY_NAME = "my_knowledge_base"
DB_PATH = "./chroma_vector_store"
HF_TOKEN = os.getenv("HF_TOKEN")

os.makedirs(DIRECTORY_NAME, exist_ok=True)

@st.cache_resource
def initialize_core_engines():
    """Loads the ML embedding layer and sets up a clean Chroma DB connection."""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Clean file locks on initial startup to avoid thread collision
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
        except Exception:
            pass
            
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(
        name="production_streamlit_robotics_manifest_v2",
        metadata={"hnsw:space": "cosine"}
    )
    return embedding_model, collection

model, hardware_collection = initialize_core_engines()

# =========================================================
# 2. DATA ACQUISITION & SEMANTIC CHUNKING
# =========================================================
@st.cache_data
def run_data_pipeline():
    """Downloads technical documents and indexes them using character-limit semantic chunking."""
    ultimate_datasets = {
        "atmega328p_register_summary.txt": "https://raw.githubusercontent.com/arduino/ArduinoCore-avr/master/cores/arduino/wiring_private.h",
        "arduino_interrupt_vectors.txt": "https://raw.githubusercontent.com/arduino/ArduinoCore-avr/master/content/handling-hardware-hardware-interrupts.md",
        "esp32_full_register_map.txt": "https://raw.githubusercontent.com/espressif/esp-idf/master/components/soc/esp32/include/soc/gpio_reg.h",
        "esp32_power_management_spec.txt": "https://raw.githubusercontent.com/espressif/esp-idf/master/components/soc/esp32/include/soc/rtc_cntl_reg.h",
        "esp32_system_clock_config.txt": "https://raw.githubusercontent.com/espressif/esp-idf/master/components/soc/esp32/include/soc/dport_reg.h",
        "mpu6050_i2c_register_addresses.txt": "https://raw.githubusercontent.com/jrowberg/i2cdevlib/master/Arduino/MPU6050/MPU6050.h",
        "spi_bus_protocol_constants.txt": "https://raw.githubusercontent.com/arduino/ArduinoCore-avr/master/libraries/SPI/src/SPI.h",
        "l298n_hbridge_driver_source.txt": "https://raw.githubusercontent.com/wobine/blackcat-matrix/master/l298n.py"
    }

    for filename, url in ultimate_datasets.items():
        dest = os.path.join(DIRECTORY_NAME, filename)
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Network timeout on repository asset: {filename}. Using local memory buffer.")

    # Generate the comprehensive data manual on disk
    with open(os.path.join(DIRECTORY_NAME, "master_silicon_datasheet.txt"), "w", encoding="utf-8") as f:
        f.write(
            "OFFICIAL EMBEDDED SYSTEMS SILICON AND ELECTRICAL DATASHEET REFERENCE MANUAL\n\n"
            "SECTION 1: ATMEGA328P (ARDUINO UNO) SILICON ARCHITECTURE\n"
            "- Absolute Maximum Ratings: DC Current per I/O Pin is 40.0 mA. Operating Voltage (VCC) electrical limit is 6.0V.\n"
            "- Hardware Registers: UART I/O Data Register maps to address UDR0. UART Control and Status Register A maps to UCSR0A.\n"
            "- Baud Rate Generator: The low byte register is UBRR0L and the high byte register is UBRR0H. Formula for Async Normal Mode: UBRR0 = (F_CPU / (16 * Baud)) - 1.\n"
            "- ADC Bit Depth: Features an integrated 6-channel 10-bit Analog-to-Digital Converter. Successive approximation conversion time takes between 13 to 26 microseconds.\n\n"
            "SECTION 2: ESP32 SYSTEM ON CHIP (SOC) ELECTROMECHANICAL SPECS\n"
            "- Maximum Per-Pin Source/Sink Current: Standard GPIO pins can source or sink a maximum of 40 mA, but the recommended operational threshold is 12 mA.\n"
            "- Flash Memory Interface: Integrates up to 16 MB of SPI flash memory tied via external memory mapping lines.\n"
            "- Power Management States: Features 5 distinct power modes: Active Mode, Modem-sleep, Light-sleep, Deep-sleep (consuming roughly 10 microamps where only the RTC timer remains active), and Hibernation Mode.\n"
            "- Sensor Sleep States: Waking up the MPU6050 IMU over I2C requires writing a clear byte 0x00 to the PWR_MGMT_1 register address 0x6B to reset the internal sleep cycle clock.\n\n"
            "SECTION 3: L298N DUAL H-BRIDGE MOTOR CONTROLLER PHYSICAL SPECS\n"
            "- Voltage Supply Limits: The maximum logic voltage limit is 7V DC. The Motor Voltage Supply (VMS) pins handle a high-power operating range from 5V DC up to 35V DC.\n"
            "- Voltage Drop Penalty: Due to internal bipolar junction transistors (BJT), the L298N suffers a severe internal voltage drop penalty. At a full 2A of output current, this internal voltage drop can be up to 4.9V, meaning a 12V motor battery power line will only supply roughly 7.1V of actual electrical potential to the physical motor terminals.\n"
            "- Total Power Dissipation: The maximum power dissipation limit is 25 Watts. Operating temperature limits range from -25 degrees Celsius up to 130 degrees Celsius. Exceeding this triggers instant thermal runaway.\n"
            "- Flyback Diode Requirement: Because DC motors are highly inductive loads, turning off a channel creates massive reverse voltage spikes (back-EMF). The circuit schematic must include 8 external fast-recovery flyback diodes (e.g., 1N4007) to shunt these voltage spikes away from the microcontroller and protect the silicon.\n"
        )

    documents_to_add, metadata_to_add, ids_to_add = [], [], []
    CHUNK_CHARACTER_LIMIT = 1000

    for filename in os.listdir(DIRECTORY_NAME):
        file_path = os.path.join(DIRECTORY_NAME, filename)
        if filename.endswith((".txt", ".md", ".ino", ".h")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            current_chunk_text = ""
            chunk_idx = 0
            
            for line in lines:
                cleaned_line = line.strip()
                if not cleaned_line:
                    continue
                    
                # Accumulate the current line payload
                current_chunk_text += cleaned_line + " "
                
                # If chunk capacity is crossed, commit it to the database stack
                if len(current_chunk_text) >= CHUNK_CHARACTER_LIMIT:
                    documents_to_add.append(current_chunk_text.strip())
                    metadata_to_add.append({"source_file": filename, "chunk_index": chunk_idx})
                    ids_to_add.append(f"semantic_chunk_{filename}_{chunk_idx}")
                    
                    current_chunk_text = ""
                    chunk_idx += 1
            
            # Flush any remaining text string trailing at the end of the file
            if len(current_chunk_text.strip()) > 15:
                documents_to_add.append(current_chunk_text.strip())
                metadata_to_add.append({"source_file": filename, "chunk_index": chunk_idx})
                ids_to_add.append(f"semantic_chunk_{filename}_{chunk_idx}")

    if documents_to_add:
        embeddings = model.encode(documents_to_add).tolist()
        hardware_collection.add(
            embeddings=embeddings, 
            documents=documents_to_add, 
            metadatas=metadata_to_add, 
            ids=ids_to_add
        )
    return hardware_collection.count()

# Initialize the vector base cleanly on startup
total_indexed_chunks = run_data_pipeline()

# =========================================================
# 3. INTERACTIVE CHAT RETRIEVAL INTERFACE
# =========================================================
def ask_copilot_engine(query):
    """Handles conversational history, reformulates shorthand queries, and executes RAG synthesis."""
    client = InferenceClient(token=HF_TOKEN)
    
    # Extract existing conversation history from Streamlit memory layers
    chat_history = st.session_state.get("chat_history", [])
    
    # --- STEP 1: QUERY REFORMULATION ---
    # If this isn't the first question, ask the LLM to rewrite it into a clean search vector
    if len(chat_history) > 0:
        history_string = ""
        for msg in chat_history[-4:]: # Read the last 4 exchanges to preserve token space
            history_string += f"{msg['role'].upper()}: {msg['content']}\n"
            
        rewrite_prompt = (
            f"Given the following conversation history and a new follow-up question, "
            f"rephrase the follow-up question into a STANDALONE engineering query that contains all necessary context. "
            f"Do not answer the question; only return the rephrased query text.\n\n"
            f"History:\n{history_string}\n"
            f"Follow-up Question: {query}\n\n"
            f"Standalone Query:"
        )
        
        try:
            rewrite_response = client.text_generation(
                prompt=rewrite_prompt,
                model="Qwen/Qwen2.5-7B-Instruct",
                max_new_tokens=100,
                temperature=0.1
            )
            search_query = rewrite_response.strip()
        except:
            search_query = query # Fallback if network drops
    else:
        search_query = query

    # --- STEP 2: HIGH-DIMENSIONAL SPATIAL RETRIEVAL ---
    query_vector = model.encode([search_query]).tolist()
    db_results = hardware_collection.query(query_embeddings=query_vector, n_results=7)
    
    if not db_results['documents'] or not db_results['documents'][0]:
        return "System error: Local storage indexes register empty.", "None", 0.0, search_query
        
    highest_score = 1.0 - db_results['distances'][0][0]
    combined_context = "\n\n---\n\n".join(db_results['documents'][0])
    source_document = db_results['metadatas'][0][0]['source_file']
    
    if highest_score < 0.45:
        return (
            "I cannot find reliable technical documentation matching that specific concept in my indexes.", 
            "None", 
            highest_score,
            search_query
        )

    # --- STEP 3: CONTEXTUAL GENERATION ---
    system_instruction = (
        "You are an expert Robotics and Embedded Systems Engineering Copilot. "
        "Answer the user's query using ONLY the verified technical datasheet context provided. "
        "Be detailed, precise, and highly technical. Maintain conversational awareness if continuing a topic. "
        "Do not invent details. If the context doesn't contain the answer, state that you do not possess the asset data."
    )
    
    # Package recent history alongside fresh context to maintain speech flow
    history_payload = ""
    for msg in chat_history[-2:]:
        history_payload += f"Past {msg['role']}: {msg['content']}\n"

    payload = f"Context from technical documents:\n{combined_context}\n\n{history_payload}\nNew Query: {query}"
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": payload}
        ],
        max_tokens=400,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip(), source_document, highest_score, search_query

# =========================================================
# 4. STATEFUL STREAMLIT CHAT LAYOUT INTERFACE
# =========================================================
st.title("🤖 Autonomous Conversational Robotics Copilot")
st.markdown("Query bare-metal systems and retain continuous contextual memory across follow-up queries.")

st.sidebar.metric(label="Indexed Knowledge Chunks", value=total_indexed_chunks)
st.sidebar.markdown("### Active Optimization Enhancements:")
st.sidebar.success("✔️ 1000-Char Semantic Chunking\n✔️ n_results=7 Context Footprint\n✔️ 0.45 Cosine Threshold Gate\n✔️ Conversational Context Memory")

# Initialize persistent message buffers if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Action button to easily wipe conversation memory arrays
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Display historical chat logs on screen with clean styling layout
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept chat queries natively from the user interface line
if user_input := st.chat_input("Ask a follow-up hardware question (e.g., What causes this penalty?):"):
    
    # Display human message instantly
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Append to memory stack
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Reconstructing contextual query vectors..."):
        ai_response, source_file, confidence, hidden_query = ask_copilot_engine(user_input)
        
    # Display AI expert response block
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        st.caption(f"📍 Source File: {source_file} | 🎯 Confidence: {confidence:.4f}")
        
        # Educational Debug Expander: Shows you the hidden ML query rewrite in action!
        with st.expander("🛠️ See Hidden ML Vector Ingestion Loop"):
            st.code(f"Original Text input: {user_input}\nMathematically Reformulated Search Query: {hidden_query}")
            
    # Append assistant's answer to memory stack
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})