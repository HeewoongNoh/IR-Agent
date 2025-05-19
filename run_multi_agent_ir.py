import sys
sys.path.append('.')
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from ir_spectrum_table import interpret_table, ir_spectrum_table_final
import json
from models.translator import *
from torch.utils.data import DataLoader
from  dataset_utils.make_data import load_dataset
from dataset_utils.scaffold_split import random_split
from evaluator_transformer_em import  MolT5Evaluator_cap2smi_em
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def topk_cosine_similarity(query: torch.Tensor, 
                                       keys: torch.Tensor, 
                                       k: int, 
                                       nist_metadata_smiles: pd.DataFrame, similarity_method:str, save_dir:str, seed: int):

    save_path = os.path.join(save_dir, f"{similarity_method}_smiles_embedding_{seed}_K_{args.k}.pt")

    if os.path.exists(save_path):
        print(f"Loading results from {save_path}")
        return torch.load(save_path)
    
    if similarity_method == "raw_spectrum":
        # Normalize
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(keys, p=2, dim=1)

        # Compute similarity
        similarity = query_norm @ keys_norm.T  # n x m

    elif similarity_method == "inner_product":
        similarity = query @ keys.T


    # Get top-k indices and values
    topk_values, topk_indices = torch.topk(similarity, k, dim=1)


    results = []
    for i in range(topk_indices.size(0)):
        smiles_similarity_map = {}
        for j in range(k):
            idx = topk_indices[i, j].item()  
            similarity_score = topk_values[i, j].item()
            smiles = nist_metadata_smiles.iloc[idx]
            smiles_similarity_map[smiles] = similarity_score
        results.append(smiles_similarity_map)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(results, save_path)
    print(f"Results saved to {save_path}")

    return results


system_prompt = "You are an expert organic chemist with specialized knowledge in analyzing infrared (IR) spectra."


load_dotenv(".env")

os.environ['OPENAI_API_KEY'] = os.environ["API_KEY"]
api_key = os.getenv("OPENAI_API_KEY")

def parse_outputs(llm_response):

    smiles_list = []
    for line in llm_response.strip().splitlines():
        match = re.search(r'\d+\.\s+(.*)', line)
        if match:
            smiles = match.group(1).strip()
            smiles_list.append(smiles)
    return smiles_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="nist")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)  
    parser.add_argument("--llm_model", type=str, default="o3-mini")   # o3-mini-medium
    parser.add_argument("--wave_len", type=int, default=4000, help="Wavenumber length")
    parser.add_argument("--max_len", type=int, default=100, help="The max length of SMILES")
    parser.add_argument("--emb_dim", type=int, default=128, help="Dimension of Embedding")
    parser.add_argument("--enc_layers", type=int, default=2, help="The number encoder layers")
    parser.add_argument("--dec_layers", type=int, default=2, help="The number decoder layers")
    parser.add_argument("--num_head", type=int, default=4, help="The number of MHA")
    parser.add_argument("--embedder", type=str, default="transformer", help="Models")
    parser.add_argument("--eval", type=int, default=1, help="evaluation step")
    parser.add_argument("--k", type=int, default=10, help="The number of retrieved SMILES of IR spectrum")
    parser.add_argument("--similarity_method", type=str, default="cosine_similarity", help="Comparing embeddings")
    parser.add_argument("--peak_mode", type=str, default='only_scale', help="Peak assign")
    parser.add_argument("--retriever_mode", type=str, default='raw_spectrum', help="smiles_pred")
    parser.add_argument("--N", type=int, default=10, help="The number of SMILES generation")
    parser.add_argument("--C", type=int, default=3, help="The number of Candidates from IR Spectra Translator")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    
    peak_mode = 'only_scale'
    peak_path = f'./dataset_utils/peak_dict.json'
    with open(peak_path, 'r') as json_file:
        peak_dict = json.load(json_file)

    # Define LLM Agents
    if args.llm_model == 'o3-mini':
        llm = ChatOpenAI(
            api_key=api_key, 
            model="o3-mini", temperature = 1.0)
    else:
        llm = ChatOpenAI(
            api_key=api_key, 
            model=args.llm_model, temperature=args.temperature
            )


    dataset = load_dataset(args.dataset, args.max_len, args.wave_len)

    dataset_train, dataset_valid, dataset_test = random_split(dataset, 0.8, 0.1, 0.1, seed=0)
    idx_train, idx_valid, idx_test = random_split(dataset, 0.8, 0.1, 0.1, seed=0, mode='index')

    # np.save('nist_train_idx.npy', idx_train)
    # np.save('nist_valid_idx.npy', idx_valid)
    # np.save('nist_test_idx.npy', idx_test)
    if args.dataset == 'nist':
        nist_metadata = pd.read_csv("")
        nist_smiles_train = nist_metadata.iloc[idx_train]['SMILES'].reset_index(drop=True)
        nist_peak_test = nist_metadata.iloc[idx_test]['id'].reset_index(drop=True)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size)

    answer_file_path = f"./dataset/smiles_pred/answer_smiles_test_seed_{args.seed}.npy"
    prediction_file_path = f"./dataset/smiles_pred/beam_10_prediction_smiles_test_seed_{args.seed}.npy"

    if os.path.exists(answer_file_path):
        loaded_predictions = np.load(answer_file_path, allow_pickle=True)
        answer_smiles_test = loaded_predictions.tolist()

    if os.path.exists(prediction_file_path):
        loaded_predictions = np.load(prediction_file_path, allow_pickle=True)
        prediction_smiles_test = loaded_predictions.tolist()

    if args.retriever_mode == 'raw_spectrum':
        save_emb_dir = "./dataset/retriever/"
        raw_spectrum_test = torch.stack([dataset_test[i][0] for i in range(len(dataset_test))])
        raw_spectrum_train = torch.stack([dataset_train[i][0] for i in range(len(dataset_train))])
        retrieved_smiles = topk_cosine_similarity(raw_spectrum_test.squeeze(2), raw_spectrum_train.squeeze(2), args.k, nist_smiles_train, args.retriever_mode, save_emb_dir, seed=0)

    else:
        print(f'wrong similarity method\n')


    class AgentState(TypedDict):
        start_prompt: str
        table_prompt: str
        table_output: str
        retriever_prompt: str
        retriever_output: str
        task_prompt: str
        task_output: str
        planner_output: str

    def Table_Agent(state: AgentState) -> AgentState:
        prompt = f"""{system_prompt}

You have an IR absorption interpretation that suggests certain substructures (e.g. nitrile, carbonyl, etc.), but this table‐based mapping can be imprecise.

Given SMILES: {given_smiles_list}
IR interpretation: {table_interpretation}

Your task is to:
For each SMILES in the given SMILES list, identify substructures that are present both in the IR interpretation and in the that SMILES.

Return a bulleted list in the format:  
substructure → confidence → brief rationale

KEEP THE RESPONSE UNDER 300 TOKENS.  
ONLY RETURN:
- A bulleted list of (substructure → confidence → brief rationale).

    """

        response = llm.invoke(input=prompt)
        state["table_prompt"] = prompt
        state["table_output"] = response.content
        return state

    def Retriever_Agent(state: AgentState) -> AgentState:

        prompt = f"""{system_prompt}

Your task is to analyze the SMILES of the candidate spectra, whose cosine similarity to the target spectrum is high.

If the target spectrum and candidate spectra exhibit high similarity, the SMILES of the target spectrum may have a similar structural characteristics to the SMILES of the candidate spectrum.

SMILES of candidate spectra and their cosine similarities to the target spectrum: {candidate_cos_similarity}

Based on the SMILES list, extract the structural information to complement the SMILES of the target spectrum.

Provide reasoning to support your analysis.

Let's think step-by-step.

KEEP THE RESPONSE UNDER 300 TOKENS.

ONLY THE REQUESTED CONTENT SHOULD BE INCLUDED IN YOUR RESPONSE.

"""

        response = llm.invoke(input=prompt)
        state["retriever_prompt"] = prompt
        state["retriever_output"] = response.content

        return state     

    def Task_Agent(state: AgentState) -> AgentState:

        prompt = f"""{system_prompt}

Your task is to refine the given SMILES list and generate a {args.N} candidate list that aligns well with the IR spectrum while preserving structural diversity and plausibility.

The IR Absorption Table Agent provides potentially useful insights by interpreting the IR spectrum and suggesting possible substructures based on known absorption patterns.

IR Spectrum Retriever Agent examines the structural features of candidate SMILES that exhibit high cosine similarity to the target spectrum.

IR Absorption Table Agent Output:
{state["table_output"]}

IR Spectrum Retriever Agent Output (high-similarity spectra & analysis):
{state["retriever_output"]}

1) Identify the substructures that are common to both the IR table interpretation and at least one SMILES in the list.

2) From the Retriever output, extract structural information (e.g., recurring motifs / scaffolds) suggested by high-similarity candidates.  

3) Guided by the structural insights from steps 1 and 2, produce a refined Top-{args.N} list of SMILES candidates.

4) Ensure the final list is chemically diverse and plausible—do not overfit to any single interpretation.

Based on these analyses, regenerate a list of Top-{args.N} SMILES by refining the target smiles: {given_smiles_list}.

Let's think step-by-step.

ONLY THE REQUESTED CONTENT SHOULD BE INCLUDED IN YOUR RESPONSE.

YOUR ANSWER FORMAT MUST BE AS FOLLOWS ONLY CONTAINING THE SMILES:
1. SMILES_1, 2. SMILES_2, 3. SMILES_3, ..., N. SMILES_N
"""

        response = llm.invoke(input=prompt)
        state["task_prompt"] = prompt
        state["task_output"] = response.content

        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("Table_Agent", Table_Agent)
    workflow.add_node("Retriever_Agent", Retriever_Agent)
    workflow.add_node("Task_Agent", Task_Agent)
    workflow.add_edge("Table_Agent", "Retriever_Agent")
    workflow.add_edge("Retriever_Agent", "Task_Agent")
    workflow.set_entry_point("Table_Agent")
    app = workflow.compile()

    logfile = open(f'./exp_results/k_{args.k}_C_{args.C}_{args.llm_model}_{args.retriever_mode}_{args.seed}.txt', 'a')


    error_cnt = 0
    smiles_refined_list, error_idx_list = [], []
    from tqdm import tqdm
    for idx, data in tqdm(nist_metadata.iloc[idx_test].reset_index(drop=True).iterrows()):
        
        try:
            ir_id = nist_peak_test.iloc[idx]
            peaks = peak_dict[ir_id]
            table_interpretation = interpret_table(peaks, ir_spectrum_table_final)
            given_smiles_list = prediction_smiles_test[idx][0:args.C]
            retrieved_smiles_prompt = {k: round(v, 4) for k, v in retrieved_smiles[idx].items()}
            candidate_cos_similarity = retrieved_smiles_prompt 
            initial_state = {"start_prompt": "Start IR Spectrum Analysis Agents"}
            print(f"idx_{idx}_new iteration starts======================================================================", file=logfile)
            for state in app.stream(initial_state):
                try:
                    if state['Table_Agent']['table_output']:
                        print(f"Table_Agent: {state['Table_Agent']['table_output']}", file=logfile)
                except KeyError:
                    pass  
                
                try:
                    if state['Retriever_Agent']['retriever_output']:
                        print(f"Retriever_Agent: {state['Retriever_Agent']['retriever_output']}", file=logfile)
                except KeyError:
                    pass  
                
                try:
                    if state['Task_Agent']['task_output']:
                        print(f"Task_Agent: {state['Task_Agent']['task_output']}", file=logfile)
                except KeyError:
                    pass  
                
                print(f'Answer SMILES:{answer_smiles_test[idx]}\n', file=logfile)
                print(f'Given SMILES:{given_smiles_list}\n', file=logfile)
                pass
            smiles_pred_results = parse_outputs(state['Task_Agent']['task_output'])
            smiles_refined_list.append(smiles_pred_results)

        except Exception as e:
            print(f"Exception raised: {e}")
            error_cnt += 1
            error_idx_list.append(idx)

    refined_file_path = f"./dataset/smiles_pred/k_{args.k}_C_{args.C}_{args.llm_model}_N_{args.N}_{args.retriever_mode}_test_seed_{args.seed}.npy"

    if os.path.exists(refined_file_path):
        print('already exists')
        pass
    else:
        np.save(refined_file_path, np.array(smiles_refined_list, dtype=object))

    evaluator = MolT5Evaluator_cap2smi_em(smiles_refined_list, answer_smiles_test)
    test_metrics = evaluator.evaluate()
    configuration = f"{args.llm_model}_seed({args.seed})_transformer_nist"
    f = open(f"./exp_results/k_{args.k}_C_{args.C}__{configuration}.txt", "a")

    print(f'Test_evaluated metrics:{test_metrics}\n')
    f.write(f'======================================================================================\n')
    f.write(f'Seed:{args.seed}\n')
    f.write(f'C:{args.C}\n')
    f.write(f'LLM_agent_model:{args.llm_model}\n')
    f.write(f'Number of retrieved SMILES:{args.k}\n')
    f.write(f'Number of generated SMILES by LLM:{args.N}\n')
    f.write(f'Retriever_mode:{args.retriever_mode}\n')
    f.write(f'Similarity mode:{args.similarity_method}\n')
    f.write(f'Number of errors raised:{error_cnt}\n')
    f.write(f'Index of errors raised:{error_idx_list}\n')
    f.write(f'Test_evaluated metrics:{test_metrics}\n')
    f.close()









