import os
import re
import json
import csv
import PyPDF2
import openai
import streamlit as st
from typing import List, Dict, Optional

# Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 4096
CHUNK_SIZE = 3000  # characters
FLASHCARDS_PER_CHUNK = 5

# Initialize OpenAI (set your API key in environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

class FlashcardGenerator:
    def __init__(self):
        self.subject_guides = {
            "General": "Generate clear, concise flashcards covering key concepts.",
            "Biology": "Focus on biological terms, processes, and relationships.",
            "History": "Emphasize dates, events, causes, and historical figures.",
            "Computer Science": "Cover algorithms, data structures, programming concepts, and definitions.",
            "Medicine": "Include anatomical terms, medical conditions, and treatments.",
            "Languages": "Create vocabulary cards with word on front, translation and example sentence on back."
        }
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text content from PDF file."""
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_content(self, content: str) -> List[str]:
        """Split content into manageable chunks for LLM processing."""
        chunks = []
        while content:
            if len(content) <= CHUNK_SIZE:
                chunks.append(content)
                break
            # Find the last space or newline within the chunk size
            split_at = content.rfind('\n', 0, CHUNK_SIZE)
            if split_at == -1:
                split_at = CHUNK_SIZE
            chunks.append(content[:split_at])
            content = content[split_at:].lstrip()
        return chunks
    
    def generate_flashcards(self, content: str, subject: str = "General") -> List[Dict]:
        """Generate flashcards from content using LLM."""
        prompt_guide = self.subject_guides.get(subject, self.subject_guides["General"])
        
        system_prompt = f"""You are an expert educational assistant that creates high-quality flashcards from educational content.
        
        {prompt_guide}
        
        Rules:
        - Generate {FLASHCARDS_PER_CHUNK} flashcards per chunk
        - Each flashcard must have a clear question and a concise answer
        - Answers should be self-contained and factually accurate
        - Include difficulty level (Easy, Medium, Hard)
        - Identify the main topic for each flashcard
        - Format as JSON with keys: question, answer, difficulty, topic
        
        Example:
        {{
            "flashcards": [
                {{
                    "question": "What is the powerhouse of the cell?",
                    "answer": "The mitochondrion is the powerhouse of the cell.",
                    "difficulty": "Easy",
                    "topic": "Cell Biology"
                }},
                {{
                    "question": "What are the three stages of cellular respiration?",
                    "answer": "The three stages are glycolysis, the Krebs cycle, and oxidative phosphorylation.",
                    "difficulty": "Medium",
                    "topic": "Cellular Respiration"
                }}
            ]
        }}"""
        
        user_prompt = f"""Please create flashcards from the following educational content:
        
        {content}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=MAX_TOKENS - len(system_prompt) - len(user_prompt)
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            # Sometimes the response includes markdown code blocks
            if '```json' in response_content:
                response_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content:
                response_content = response_content.split('```')[1]
            
            flashcards = json.loads(response_content)["flashcards"]
            return flashcards
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")
            return []
    
    def process_content(self, content: str, subject: str = "General") -> List[Dict]:
        """Process entire content by chunking and generating flashcards."""
        chunks = self.chunk_content(content)
        all_flashcards = []
        
        with st.spinner(f"Generating flashcards from {len(chunks)} sections..."):
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                flashcards = self.generate_flashcards(chunk, subject)
                all_flashcards.extend(flashcards)
                progress_bar.progress((i + 1) / len(chunks))
        
        return all_flashcards
    
    def group_by_topic(self, flashcards: List[Dict]) -> Dict[str, List[Dict]]:
        """Group flashcards by their detected topic."""
        topics = {}
        for card in flashcards:
            topic = card.get("topic", "Uncategorized")
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(card)
        return topics
    
    def export_csv(self, flashcards: List[Dict], filename: str = "flashcards.csv"):
        """Export flashcards to CSV file."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer', 'difficulty', 'topic']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for card in flashcards:
                writer.writerow(card)
        return filename
    
    def export_json(self, flashcards: List[Dict], filename: str = "flashcards.json"):
        """Export flashcards to JSON file."""
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump({"flashcards": flashcards}, jsonfile, indent=2)
        return filename
    
    def export_anki(self, flashcards: List[Dict], filename: str = "flashcards_anki.txt"):
        """Export flashcards to Anki-compatible TSV format."""
        with open(filename, 'w', encoding='utf-8') as ankifile:
            for card in flashcards:
                # Anki format: front\tback\ttags
                tags = f"{card['difficulty']} {card['topic']}".replace(" ", "_")
                line = f"{card['question']}\t{card['answer']}\t{tags}\n"
                ankifile.write(line)
        return filename

def main():
    st.set_page_config(page_title="LLM Flashcard Generator", page_icon="📚")
    st.title("📚 LLM-Powered Flashcard Generator")
    st.write("Convert educational content into flashcards using AI")
    
    generator = FlashcardGenerator()
    
    # Input options
    input_method = st.radio("Input method:", ("Text input", "File upload"))
    
    content = ""
    if input_method == "Text input":
        content = st.text_area("Paste your educational content here:", height=300)
    else:
        uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                content = generator.extract_text_from_pdf(uploaded_file)
            else:
                content = uploaded_file.read().decode("utf-8")
    
    # Subject selection
    subject = st.selectbox(
        "Select subject area (helps tailor flashcards):",
        list(generator.subject_guides.keys())
    )
    
    if st.button("Generate Flashcards") and content:
        flashcards = generator.process_content(content, subject)
        
        if not flashcards:
            st.error("No flashcards were generated. Please try with different content.")
            return
        
        st.success(f"Generated {len(flashcards)} flashcards!")
        
        # Group by topic
        topics = generator.group_by_topic(flashcards)
        
        # Display flashcards by topic with editing capability
        edited_flashcards = []
        for topic, cards in topics.items():
            with st.expander(f"Topic: {topic} ({len(cards)} cards)"):
                for i, card in enumerate(cards):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_question = st.text_input(
                            f"Question {i+1}", 
                            value=card["question"],
                            key=f"q_{topic}_{i}"
                        )
                    with col2:
                        new_answer = st.text_input(
                            f"Answer {i+1}", 
                            value=card["answer"],
                            key=f"a_{topic}_{i}"
                        )
                    col3, col4, _ = st.columns([1, 1, 2])
                    with col3:
                        new_difficulty = st.selectbox(
                            "Difficulty",
                            ["Easy", "Medium", "Hard"],
                            index=["Easy", "Medium", "Hard"].index(card["difficulty"]),
                            key=f"d_{topic}_{i}"
                        )
                    with col4:
                        new_topic = st.text_input(
                            "Topic",
                            value=card["topic"],
                            key=f"t_{topic}_{i}"
                        )
                    
                    edited_flashcards.append({
                        "question": new_question,
                        "answer": new_answer,
                        "difficulty": new_difficulty,
                        "topic": new_topic
                    })
        
        # Export options
        st.subheader("Export Flashcards")
        export_format = st.selectbox("Select export format:", ["CSV", "JSON", "Anki"])
        
        if st.button("Export"):
            if not edited_flashcards:
                edited_flashcards = flashcards
            
            export_filename = f"flashcards_{subject.lower()}.{export_format.lower()}"
            if export_format == "CSV":
                filename = generator.export_csv(edited_flashcards, export_filename)
            elif export_format == "JSON":
                filename = generator.export_json(edited_flashcards, export_filename)
            else:  # Anki
                filename = generator.export_anki(edited_flashcards, export_filename)
            
            with open(filename, "rb") as f:
                st.download_button(
                    label="Download Export File",
                    data=f,
                    file_name=filename,
                    mime="text/csv" if export_format == "CSV" else "application/json"
                )

if __name__ == "__main__":
    main()
