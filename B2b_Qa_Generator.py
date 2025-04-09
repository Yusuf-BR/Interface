# ✅ Reusable Python Function: Dynamic Q&A Generator for Fine-tuning + RAG (Clean Version for Streamlit App)

import pandas as pd
import json

# === Strategic recommendation logic with segmentation ===
def strategic_recommendation(value, global_avg):
    if value > global_avg * 1.2:
        return "This is a hot lead — engage immediately to maximize impact. Success rates here are well above average, making it a prime target."
    elif value > global_avg * 1.05:
        return "This is a warm lead, suitable for nurturing through follow-up campaigns. Success rates are above average, offering good potential."
    elif value >= global_avg * 0.95:
        return "This is a warm lead with moderate potential, monitor and optimize your outreach. Success rates are in line with the overall average."
    elif value >= global_avg * 0.8:
        return "This is a cooler lead, consider refining your targeting to uncover hidden opportunities. Success rates are slightly below average."
    else:
        return "This is a cold lead, better to deprioritize for now and re-evaluate later. Success rates are below average, indicating limited return on investment."

# === Main Q&A generation function ===
def generate_b2b_qa(input_csv_path, output_json_path):
    df = pd.read_csv(input_csv_path)
    qa_pairs = []
    avg_conversion = df['Conversion Probability'].mean()
    potential_threshold = df['Conversion Probability'].quantile(0.75)

    def add_variations(question_base, answer_base):
        variations = [
            question_base,
            question_base.replace("potential", "prospective leads"),
            question_base.replace("potential", "opportunities"),
            question_base.replace("What is", "Tell me about"),
            question_base.replace("What is", "How about"),
            question_base.replace("What is", "Can you explain"),
            question_base.replace("How many", "Could you tell me how many"),
            question_base.replace("How many", "Do you know how many"),
        ]
        unique_variations = list(set(variations))
        for q in unique_variations:
            qa_pairs.append({
                "instruction": q,
                "input": "",
                "output": answer_base
            })

    # Individual clients analysis with success perspective
    for _, row in df.iterrows():
        recommendation = strategic_recommendation(row['Conversion Probability'], avg_conversion)

        base_answer = f"{row['Name']} from {row['Company']} has a conversion probability of {row['Conversion Probability']:.4f}. {recommendation}"
        add_variations(f"What is the conversion probability of {row['Name']}?", base_answer)

        base_answer = f"{row['Name']} is currently working at {row['Company']}. Use this knowledge to build a targeted and relevant outreach strategy."
        add_variations(f"Which company does {row['Name']} work for?", base_answer)
        add_variations(f"In which company is {row['Name']} employed?", base_answer)
        add_variations(f"What company is {row['Name']} affiliated with?", base_answer)
        add_variations(f"Can you tell me the company of {row['Name']}?", base_answer)

        base_answer = f"{row['Name']} holds the position of {row['Job title']} at {row['Company']}. Tailor your messaging to address the specific priorities of this role and company."
        add_variations(f"What is the job title of {row['Name']}?", base_answer)

        base_answer = f"You can reach {row['Name']} directly via LinkedIn: {row['Links']}. Use this for personalized engagement."
        add_variations(f"What is the LinkedIn profile of {row['Name']}?", base_answer)

        base_answer = f"{row['Company']} employs {row['Number of employees']} people. Adjust your proposals to reflect the scale and operational complexity of this organization."
        add_variations(f"How many employees does {row['Company']} have?", base_answer)

        base_answer = f"{row['Company']} operates in the {row['Industry']} industry. Adapt your approach to align with industry expectations and trends for better engagement."
        add_variations(f"What is the industry of {row['Company']}?", base_answer)

    # Combined analysis: Region + Industry + Job title
    for region in df['Region'].dropna().unique():
        for industry in df['Industry'].dropna().unique():
            for title in df['Job title'].dropna().unique():
                combined_df = df[(df['Region'] == region) & (df['Industry'] == industry) & (df['Job title'] == title)]
                if not combined_df.empty:
                    potential = combined_df[combined_df['Conversion Probability'] > avg_conversion]
                    combined_avg = combined_df['Conversion Probability'].mean()
                    recommendation = strategic_recommendation(combined_avg, avg_conversion)
                    base_answer = f"In {region}, within the {industry} industry, professionals with the title '{title}' present {len(potential)} opportunities, averaging a conversion rate of {combined_avg:.4f}. {recommendation}"
                    add_variations(f"For the job title '{title}' in the {industry} industry in {region}, what is the potential?", base_answer)

    # === Export clean JSON file ===
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ Q&A generated successfully! File saved as: {output_json_path}")

# ✅ Ready for integration into Streamlit app!
# Example usage:
# generate_b2b_qa('Filtered_and_Sampled_Data.csv', 'b2b_dynamic_qa.json')



