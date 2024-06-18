import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from clips import Environment, Symbol
from flask import Flask, render_template, request

app = Flask(__name__)

class Question1:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def disease_symptom_mapping(self):
        disease_to_symptoms = {}
        for index, row in self.data.iterrows():
            disease = row['Disease']
            symptoms = set(
                row[symptom] for symptom in self.data.columns if 'Symptom' in symptom and pd.notna(row[symptom]))
            if disease not in disease_to_symptoms:
                disease_to_symptoms[disease] = symptoms
            else:
                disease_to_symptoms[disease].update(symptoms)
        return disease_to_symptoms

    def symptom_disease_mapping(self):
        symptom_to_diseases = {}
        for index, row in self.data.iterrows():
            disease = row['Disease']
            for symptom in self.data.columns:
                if 'Symptom' in symptom and pd.notna(row[symptom]):
                    if row[symptom] not in symptom_to_diseases:
                        symptom_to_diseases[row[symptom]] = {disease}
                    else:
                        symptom_to_diseases[row[symptom]].add(disease)
        return symptom_to_diseases

    def plot_knowledge_graph(self, mapping):
        G = nx.Graph()
        for key, values in mapping.items():
            G.add_node(key, bipartite=0)  # Disease node
            for value in values:
                G.add_node(value, bipartite=1)  # Symptom node
                G.add_edge(key, value)

        pos = nx.spring_layout(G, k=0.1, iterations=20)  # Regulates the distance between nodes

        plt.figure(figsize=(20, 20))  # Increases the figure size
        nx.draw(G, pos, node_size=10, width=0.5, font_size=8, with_labels=True, alpha=0.7, edge_color='b')

        plt.axis('off')
        plt.show()

class Question2:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        description_path = dataset_path.replace('dataset.csv', 'symptom_Description.csv')
        precaution_path = dataset_path.replace('dataset.csv', 'symptom_precaution.csv')
        self.symptom_description = pd.read_csv(description_path)
        self.symptom_precaution = pd.read_csv(precaution_path)

    def create_clips_rules(self):
        combined_data = self.data.merge(self.symptom_description, on='Disease', how='left').merge(
            self.symptom_precaution, on='Disease', how='left')

        rule_counter = {}
        rules = []
        for _, row in combined_data.iterrows():
            disease = row['Disease'].replace(" ", "_").replace("-", "_").replace("'", "\\'").replace("/", "_")

            # Initializes or increments the rule counter for the diseases
            if disease not in rule_counter:
                rule_counter[disease] = 1
            else:
                rule_counter[disease] += 1


            rule_name = f"{disease}_{rule_counter[disease]}"

            symptoms = [
                f"(symptom {row[f'Symptom_{i}'].strip().replace(' ', '_').replace('-', '_').replace("'", "\\'").replace('/', '_')})"
                for i in range(1, 18)
                if pd.notna(row.get(f'Symptom_{i}'))
            ]
            if not symptoms:
                continue  # Skip if no symptoms

            symptoms_clause = ' '.join(symptoms)
            if len(symptoms) > 1:
                symptoms_clause = f"(or {symptoms_clause})"

            description = row['Description'] if pd.notna(row['Description']) else "No description available"
            description = description.replace('"', '\\"').replace("\n", " ")

            precautions = ', '.join([p.strip().replace(' ', '_').replace('-', '_').replace('"', '\\"').replace('/', '_')
                                     for p in [row.get(f'Precaution_{i}') for i in range(1, 5)] if pd.notna(p)])

            rule = f'(defrule {rule_name}\n' \
                   f'   {symptoms_clause}\n' \
                   f'=>\n' \
                   f'   (assert (diagnosis "{disease}"\n' \
                   f'       description "{description}"\n' \
                   f'       precautions "{precautions}"))\n'
            rules.append(rule)

        return rules

    def write_rules_to_file(self, rules, file_path):
        with open(file_path, 'w') as file:
            for rule in rules:
                file.write(rule + '\n\n')


def load_and_test_clips(file_path, symptoms):
    env = Environment()
    env.load(file_path)
    env.reset()


    for symptom in symptoms:
        env.assert_string(f'(symptom {symptom})')

    # Run the inference engine
    env.run()




class Question3:
    def __init__(self, data_path, symptom_to_diseases):
        self.data = pd.read_csv(data_path)
        self.description = pd.read_csv(data_path.replace('dataset.csv', 'symptom_Description.csv'))
        self.precaution = pd.read_csv(data_path.replace('dataset.csv', 'symptom_precaution.csv'))
        self.symptom_severity = pd.read_csv(data_path.replace('dataset.csv', 'Symptom-severity.csv'))
        self.symptoms_to_diseases = symptom_to_diseases
        self.symptoms = self.load_symptoms()

    def load_symptoms(self):
        symptoms = set()
        for i in range(1, 18):
            symptoms.update(self.data[f'Symptom_{i}'].dropna().unique())
        return list(symptoms)

    def find_diseases(self, selected_symptoms):
        disease_matches = {}
        symptom_severity = {row['Symptom']: row['weight'] for index, row in self.symptom_severity.iterrows()}

        for symptom in selected_symptoms:
            if symptom in self.symptoms_to_diseases:
                for disease in self.symptoms_to_diseases[symptom]:
                    if disease not in disease_matches:
                        disease_matches[disease] = {
                            'count': 0,
                            'symptoms': set(),
                            'severity_score': 0,
                            'description': "No description available",
                            'precautions': []
                        }
                    disease_matches[disease]['count'] += 1
                    disease_matches[disease]['symptoms'].add(symptom)
                    disease_matches[disease]['severity_score'] += symptom_severity.get(symptom, 0)

        # Sorts diseases by the severity score and the number of matched symptoms
        sorted_diseases = sorted(disease_matches.items(), key=lambda x: (x[1]['severity_score'], x[1]['count']), reverse=True)

        results = []
        for disease, info in sorted_diseases:
            desc = self.description.loc[self.description['Disease'] == disease, 'Description'].values[0] if not \
                self.description.loc[self.description['Disease'] == disease, 'Description'].empty else info['description']
            precautions_list = []
            for i in range(1, 5):
                precaution_column = f'Precaution_{i}'
                precaution_value = self.precaution.loc[self.precaution['Disease'] == disease, precaution_column]
                if not precaution_value.empty:
                    precautions_list.append(precaution_value.values[0])

            results.append({
                'Disease': disease,
                'Description': desc,
                'Precautions': precautions_list,
                'Symptom Match Count': info['count'],
                'Matched Symptoms': list(info['symptoms']),
                'Severity Score': info['severity_score']
            })

        return results

@app.route('/')
def index():
    symptoms = question3.symptoms
    return render_template('index.html', symptoms=symptoms)

@app.route('/results', methods=['POST'])
def results():
    selected_symptoms = request.form.getlist('symptoms')
    results = question3.find_diseases(selected_symptoms)
    return render_template('results.html', results=results)

def main():
    # Path to dataset file
    file_path = 'dataset.csv'

    q1 = Question1(file_path)

    # Mappings
    disease_to_symptoms = q1.disease_symptom_mapping()
    symptom_to_diseases = q1.symptom_disease_mapping()

    # Print mappings to verify
    print("Disease to Symptoms Mapping:", disease_to_symptoms)
    print("Symptom to Diseases Mapping:", symptom_to_diseases)

    # Plot mapping
    q1.plot_knowledge_graph(disease_to_symptoms)

    q2 = Question2(file_path)

    q2 = Question2(file_path)
    rules = q2.create_clips_rules()
    clp_file_path = 'medical_knowledge_base.clp'
    q2.write_rules_to_file(rules, clp_file_path)
    # List of symptoms to test
    test_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
   # load_and_test_clips(clp_file_path, test_symptoms)

    global question3
    question3 = Question3(file_path,symptom_to_diseases)
    app.run(debug=True)

if __name__ == "__main__":
    main()