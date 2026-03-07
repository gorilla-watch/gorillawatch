import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
from collections import Counter
import torch
import fiftyone as fo
import fiftyone.brain as fob
from datetime import datetime
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE

from gorillawatch.model.basemodel import TimmWrapper
from gorillawatch.model.model_evaluation import generate_embeddings, calculate_distance
from gorillawatch.data.data_loading import get_image_paths

from gorillawatch.args import *

EVALUATR_RESULTS_PATH = "/workspaces/gorillawatch/src/gorillawatch/qualitative_evaluate/evaluation_results"

# KNN variation: return y_true, y_pred, class_names, and confidence scores
def KNN_qual_eval(query_embeddings, images_to_check, query_labels, query_video_ids, config, gallery_loader, model, distance_metric="euclidean", num_neighbors=K):

    # set up queries for KNN
    query_embeddings = query_embeddings.to(DEVICE)
    query_labels_tensor = torch.as_tensor(query_labels, device=DEVICE)
    query_video_ids_tensor = torch.as_tensor(query_video_ids, device=DEVICE)

    # set up gallery (database) for KNN retrieval
    gallery_embeddings, gallery_labels, gallery_video_ids = generate_embeddings(model, gallery_loader, config, use_amp=True)
    embeddings = gallery_embeddings.to(DEVICE)
    labels_tensor = torch.as_tensor(gallery_labels, device=DEVICE)
    video_ids_tensor = torch.as_tensor(gallery_video_ids, device=DEVICE)

    print("Number of queries: ", len(images_to_check))
    print("Number of gallery embeddings: ", len(embeddings))

    num_predictions = len(images_to_check)
    predictions = torch.zeros(num_predictions, dtype=torch.long, device=DEVICE)
    actuals = torch.zeros(num_predictions, dtype=torch.long, device=DEVICE)
    confidence_scores = torch.zeros(num_predictions, dtype=torch.float, device=DEVICE)

    # iterate through each query embedding against the gallery embeddings
    for i, idx in enumerate(images_to_check):

        test_embedding = query_embeddings[idx].unsqueeze(0)
        distances = calculate_distance(embeddings, test_embedding, distance_metric)

        # Mask out self and same video indices
        mask = torch.ones_like(distances, dtype=torch.bool, device=DEVICE)
        mask[idx] = False  # Exclude self
        mask &= (video_ids_tensor != query_video_ids_tensor[idx])  # Different videos

        # Get top k neighbor indices
        top_k_values, top_k_indices = torch.topk(distances[mask], k=min(num_neighbors, mask.sum().item()), largest=False)
        neighbor_indices = torch.nonzero(mask)[top_k_indices].squeeze(1)

        if len(neighbor_indices) > 0:
            neighbor_labels = labels_tensor[neighbor_indices]
            # Convert to integers for bincount
            neighbor_labels = neighbor_labels.long()
            label_counts = torch.bincount(neighbor_labels)
            predicted_label = torch.argmax(label_counts)
            predictions[i] = predicted_label
            actuals[i] = query_labels_tensor[idx]

            confidence_scores[i] = torch.sum(neighbor_labels == predicted_label).float() / len(neighbor_labels)
            
    
    accuracy = (predictions == actuals).float().mean().item()
    print("correct predictions: ", accuracy * len(predictions))
    
    return embeddings.cpu().numpy(), actuals.cpu().numpy(), predictions.cpu().numpy(), confidence_scores.cpu().numpy()

# This module provides a comprehensive evaluation tool for gorilla re-identification models.
class GorillaReIDEvaluator:
    """
    A comprehensive evaluation tool for gorilla re-identification models
    that takes class imbalance into account.
    """
    
    def __init__(self, y_true, y_pred, pred_scores, label_names, locations=None):
        """
        Initialize the evaluator with ground truth and predictions.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth labels
        y_pred : array-like
            Predicted labels
        pred_scores : array-like, optional
            Prediction confidence scores for ROC analysis
        label_names : list
            List of class names
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.pred_scores = pred_scores
        self.class_names = label_names
        self.locations = locations
            
        # Calculate class distribution
        self.class_distribution = Counter(y_true)
        self.total_samples = len(y_true)
        
        # calculate social groups
        self.social_groups = {}
        for class_name in self.class_names:
            if len(class_name) >= 2:
                group = class_name[:2]
                if group not in self.social_groups:
                    self.social_groups[group] = []
                self.social_groups[group].append(class_name)
        
        # calculate gender groups
        self.gender_groups = {
            'silverback': [],
            'adult_female': [],
            'blackback': [],
            'adolescent': [],
            'infant': []
        }

        for class_name in self.class_names:
            idx = int(class_name[2:])
            
            if idx == 0:
                category = 'silverback'
            elif 1 <= idx <= 19:
                category = 'adult_female'
            elif 20 <= idx <= 39:
                category = 'blackback'
            elif 40 <= idx <= 59:
                category = 'adolescent'
            elif 60 <= idx <= 79:
                category = 'infant'
            else:
                continue  # Skip if idx doesn't match any category
                
            self.gender_groups[category].append(class_name)
         
        # calculate locations
        if self.locations is not None:
            self.location_groups = {}
            for y_true, y_pred, location in zip(self.y_true, self.y_pred, self.locations):
                if location not in self.location_groups:
                    self.location_groups[location] = []
                self.location_groups[location].append((y_true, y_pred))
        
    def basic_metrics(self):
        """Calculate and return basic classification metrics."""
        accuracy = np.mean(self.y_true == self.y_pred)
        balanced_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        kappa = cohen_kappa_score(self.y_true, self.y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        
        class_metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Per location metrics
        if self.location_groups is not None:
            location_metrics = {}
            for location, predictions in self.location_groups.items():
                y_true_loc = [p[0] for p in predictions]
                y_pred_loc = [p[1] for p in predictions]
                loc_accuracy = accuracy_score(y_true_loc, y_pred_loc)
                loc_balanced_acc = balanced_accuracy_score(y_true_loc, y_pred_loc)

                # Get the unique classes at this location
                unique_individuals_idx = sorted(set(y_true_loc))
                unique_individuals = [self.class_names[i] for i in unique_individuals_idx]
                individual_counts = {self.class_names[cls]: y_true_loc.count(cls) for cls in unique_individuals_idx}
                
                location_metrics[location] = {
                    'Location': location,
                    'Accuracy': loc_accuracy,
                    'Balanced Accuracy': loc_balanced_acc,
                    'Individuals': unique_individuals,
                    'Individual Counts': individual_counts
                }
            
            location_metrics_df = pd.DataFrame.from_dict(location_metrics, orient='index')
            
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )
    
        summary = {
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'Cohen\'s Kappa': kappa,
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1': macro_f1,
            'Weighted Precision': weighted_precision,
            'Weighted Recall': weighted_recall,
            'Weighted F1': weighted_f1
        }
        
        return class_metrics_df, location_metrics_df, pd.Series(summary)
    
    def class_distribution_analysis(self):
        """Analyze and visualize the class distribution."""
        # Create DataFrame for class distribution
        dist_df = pd.DataFrame({
            'Class': self.class_names,
            'Count': [self.class_distribution.get(i, 0) for i in range(len(self.class_names))],
        })
        dist_df['Percentage'] = dist_df['Count'] / self.total_samples * 100
        dist_df = dist_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        return dist_df
    
    def plot_class_distribution(self):
        """Plot the class distribution."""
        dist_df = self.class_distribution_analysis()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Count', data=dist_df)
        plt.title('Class Distribution')
        plt.xlabel('Gorilla Individual')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add percentages on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 5,
                    f'{dist_df["Percentage"].iloc[i]:.1f}%',
                    ha="center") 
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_confusion_matrix(self, normalize=True):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def error_analysis(self):
        """Analyze the misclassified samples."""
        errors = self.y_true != self.y_pred
        error_indices = np.where(errors)[0]
        
        error_df = pd.DataFrame({
            'Index': error_indices,
            'True_Class': self.y_true[error_indices],
            'Predicted_Class': self.y_pred[error_indices]
        })
        
        if self.class_names is not None:
            error_df['True_Class_Name'] = [self.class_names[int(c)] for c in error_df['True_Class']]
            error_df['Predicted_Class_Name'] = [self.class_names[int(c)] for c in error_df['Predicted_Class']]
        
        # Summarize errors by class
        error_summary = {}
        for true_class in np.unique(self.y_true):
            class_indices = np.where(self.y_true == true_class)[0]
            class_errors = np.sum(self.y_pred[class_indices] != true_class)
            error_rate = class_errors / len(class_indices) if len(class_indices) > 0 else 0
            class_key = self.class_names[true_class] if self.class_names is not None else str(true_class)
            error_summary[class_key] = {
                'Total_Samples': len(class_indices),
                'Misclassified': class_errors,
                'Error_Rate': error_rate
            }
        
        # Convert to DataFrame and sort by Error_Rate in descending order
        error_summary_df = pd.DataFrame(error_summary).T
        error_summary_df = error_summary_df.sort_values(by='Error_Rate', ascending=False)
        
        # Optionally, sort the error_df by True_Class based on error rate
        if not error_df.empty:
            error_rate_by_class = {k: v['Error_Rate'] for k, v in error_summary.items()}
            error_df['Error_Rate'] = error_df['True_Class'].map(error_rate_by_class)
            error_df = error_df.sort_values(by='Error_Rate', ascending=False)
            # Remove the temporary column if not needed
            error_df = error_df.drop(columns=['Error_Rate'])
        
        return error_df, error_summary_df
    
    def plot_error_rates(self):
        """Plot the error rates by class."""
        _, error_summary = self.error_analysis()
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=error_summary.index, y=error_summary['Error_Rate'])
        plt.title('Error Rate by Class')
        plt.xlabel('Gorilla Individual')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)
        
        # Add percentages on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.02,
                    f'{height:.1%}',
                    ha="center") 
        
        plt.tight_layout()
        return plt.gcf()
    
    def class_similarity_analysis(self):
        """
        Analyze which classes are most commonly confused with each other.
        Returns a similarity matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        np.fill_diagonal(cm, 0)  # Zero out diagonal to focus on misclassifications
        
        # Normalize by row totals
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Handle division by zero
        cm_norm = np.nan_to_num(cm_norm)
        
        return pd.DataFrame(cm_norm, index=self.class_names, columns=self.class_names)
    
    def plot_class_similarity(self):
        """Plot the class similarity matrix."""
        similarity = self.class_similarity_analysis()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity, annot=True, fmt='.2f', cmap='Reds')
        plt.title('Class Similarity Matrix (Misclassification Rates)')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        return plt.gcf()
    
    def find_most_confused_pairs(self, top_n=10):
        """
        Find the most commonly confused pairs of gorilla individuals.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top confused pairs to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the top confused pairs with confusion rates
        """
        # Get the similarity matrix from existing method
        similarity_matrix = self.class_similarity_analysis()
        
        # Convert to long format for easier sorting
        confusion_pairs = []
        
        # Get the raw confusion matrix for counts
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Iterate through the similarity matrix
        for true_class in similarity_matrix.index:
            for pred_class in similarity_matrix.columns:
                if true_class != pred_class:  # Skip diagonal
                    # Find the indices to get the raw count
                    true_idx = list(similarity_matrix.index).index(true_class)
                    pred_idx = list(similarity_matrix.columns).index(pred_class)
                    
                    # Calculate counts and rate
                    count = cm[true_idx, pred_idx]
                    support = cm[true_idx].sum()
                    confusion_rate = count / support if support > 0 else 0
                    
                    if confusion_rate > 0:  # Only include non-zero confusions
                        confusion_pairs.append({
                            'True_Class': true_class,
                            'Predicted_Class': pred_class,
                            'Confusion_Rate': confusion_rate,
                            'Percentage': f"{confusion_rate * 100:.2f}%",
                            'Confusion_Count': int(count),
                            'Support': int(support)
                        })
        
        # Convert to DataFrame
        confusion_df = pd.DataFrame(confusion_pairs)
        
        # Sort by confusion rate (descending)
        sorted_df = confusion_df.sort_values(by='Confusion_Rate', ascending=False).reset_index(drop=True)
        
        # Return top N confused pairs
        return sorted_df.head(top_n)

    def per_group_confusion_analysis(self, n_groups=None, group_type='social group'):
        """
        Analyze confusion patterns within social groups of gorillas.
        Each gorilla's social group is identified by the first two letters of its label.
        
        Parameters:
        -----------
        n_groups : int, optional
            Number of top confused social groups to return (by default returns all)
        
        Returns:
        --------
        dict
            Dictionary with social group names as keys and DataFrames of confusion rates as values
                'confusion_matrix': confusion matrix with raw counts, 
                'confusion_rate': confusion matrix with rates of missclassification,
                'error_distribution': percentage of errors within the group out of all errors,
                'members': members,
                'overall_rate': overall confusion rate of the group (total missclassifications within the group / total missclassification within and outside the group)
        """
        
        # get the corrsponding group
        if group_type == 'social group':
            chosen_groups = self.social_groups
        elif group_type == 'gender group':
            chosen_groups = self.gender_groups
        
        # Get the full confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        # Analyze each social group
        group_results = {}
        group_confusion_rates = {}
        
        for group, members in chosen_groups.items():
            if len(members) <= 1:
                continue  # Skip groups with only one individual
                
            # Extract the sub-matrix for this group
            group_cm = cm_df.loc[members, members].copy()
            
            # Zero out diagonal to focus on misclassifications
            for member in members:
                group_cm.loc[member, member] = 0
                
            # Calculate confusion rates within group
            row_sums = group_cm.sum(axis=1)
            full_row_sums = cm_df.loc[members].sum(axis=1)
            
            # For each member, what percentage of its errors are within the group (missclassifications within the group / all misclassifications)
            within_group_error_rate = row_sums / (full_row_sums - np.diag(cm_df.loc[members, members]))
            
            # For each member, what percentage of all its instances are misclassified within the group (miss classifications within the group / all instances of the member)
            within_group_confusion_rate = row_sums / cm_df.loc[members].sum(axis=1)
            if isinstance(within_group_error_rate, pd.Series):
                within_group_error_rate = within_group_error_rate.to_frame()
            # Calculate overall group confusion rate (how frequently members are confused with each other)
            total_within_group_confusions = row_sums.sum()
            total_group_samples = full_row_sums.sum()
            total_correct_classifications = sum(cm_df.loc[member, member] for member in members)
            
            group_confusion_rate = total_within_group_confusions / (total_group_samples - total_correct_classifications) if (total_group_samples - total_correct_classifications) > 0 else 0
                        
            # Store normalized confusion matrix and overall rate
            group_results[group] = {
                'confusion_matrix': pd.DataFrame(group_cm, index=members, columns=members),
                'confusion_rate': group_cm.div(full_row_sums, axis=0).fillna(0),
                'error_distribution': within_group_error_rate.fillna(0),
                'members': members,
                'overall_rate': group_confusion_rate
            }
            
            group_confusion_rates[group] = group_confusion_rate
        
        # Sort groups by confusion rate if n_groups is specified
        if n_groups is not None:
            top_groups = sorted(group_confusion_rates.items(), key=lambda x: x[1], reverse=True)[:n_groups]
            filtered_results = {group: group_results[group] for group, _ in top_groups}
            return filtered_results
        
        return group_results
    
    def plot_group_confusion(self, group_type='social group', selected_group=None, metric='confusion_rate', figsize=(15, 12)):
        """
        Plot the confusion matrix for a specific social group or all groups.
        
        Parameters:
        -----------
        selected_group : str, optional
            Social group to plot. If None, plots all groups.
        metric : str, optional
            Which metric to visualize: 'confusion_rate' or 'error_distribution'
        figsize : tuple, optional
            Figure size for the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        # Get the social group analysis
        # group_results = self.social_group_confusion_analysis()
        
        if group_type == 'social group':
            group_results = self.per_group_confusion_analysis(group_type='social group')
        elif group_type == 'gender group':
            group_results = self.per_group_confusion_analysis(group_type='gender group')
        
        # Filter out groups with only one member
        valid_groups = {group: data for group, data in group_results.items() 
                    if len(data['members']) > 1}
        
        if not valid_groups:
            print("No valid social groups found with multiple members.")
            return None
        
            
        # If a specific group is selected, plot just that group
        if selected_group is not None:
            if selected_group not in valid_groups:
                print(f"Group {selected_group} not found or has fewer than 2 members.")
                print(f"Available groups: {list(valid_groups.keys())}")
                return None
                
            group_data = valid_groups[selected_group]
            if metric == 'confusion_rate':
                plot_data = group_data['confusion_rate']
                title = f"Within-Group Confusion Rates for {selected_group} Group"
                label = "% of samples misclassified"
            elif metric == 'error_distribution':
                plot_data = group_data['error_distribution']
                title = f"Error Distribution Within {selected_group} Group"
                label = "% of errors within group"
            else:
                print(f"Invalid metric: {metric}. Use 'confusion_rate' or 'error_distribution'.")
                return None
            
            plt.figure(figsize=figsize)
            sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='Reds', 
                        linewidths=0.5, cbar_kws={'label': label})
            plt.title(title)
            plt.xlabel('Predicted Gorilla')
            plt.ylabel('True Gorilla')
            plt.suptitle(f"Overall within-group confusion rate: {group_data['overall_rate']:.2f}", 
                        fontsize=10, y=0.92)
            plt.tight_layout()
            return plt.gcf()
        
        # If no specific group is selected, plot all groups
        else:
            # Sort groups by their overall confusion rate
            sorted_groups = sorted(valid_groups.items(), 
                                key=lambda x: x[1]['overall_rate'], 
                                reverse=True)
            
            # Calculate grid dimensions
            n_groups = len(sorted_groups)
            n_cols = math.ceil((math.sqrt(n_groups)))
            n_rows = (n_groups + n_cols - 1) // n_cols  # Ceiling division
            
            # Create subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Plot each group
            for i, (group, data) in enumerate(sorted_groups):
                if i < len(axes):
                    ax = axes[i]
                    
                    if metric == 'confusion_rate':
                        plot_data = data['confusion_rate']
                        title = f"{group} Group"
                        label = "% of samples"
                    else:
                        plot_data = data['error_distribution']
                        title = f"{group} Group"
                        label = "% of errors"
                    
                    sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='Reds', 
                            linewidths=0.5, ax=ax, cbar_kws={'label': label})
                    ax.set_title(f"{title} (rate: {data['overall_rate']:.2f})")
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
            
            # Hide any unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle(f"Group Confusion Analysis ({group_type}) - {metric.replace('_', ' ').title()}", 
                        fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            return fig
    

    def generate_full_report(self, output_dir=None):
        """
        Generate a comprehensive evaluation report, including all metrics and plots.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the report. If None, report is displayed but not saved.
        """
        # Get basic metrics
        class_metrics_df, location_metrics_df, summary = self.basic_metrics()
        print("=== Basic Metrics ===")
        print(summary)
        print("\n=== Per-Class Metrics ===")
        print(class_metrics_df)
        print("\n=== Per-Location Metrics ===")
        print(location_metrics_df)
        
        # Plot class distribution
        print("\n=== Class Distribution ===")
        dist_df = self.class_distribution_analysis()
        print(dist_df)
        fig1 = self.plot_class_distribution()
        
        # Plot confusion matrix
        print("\n=== Confusion Matrix ===")
        fig2 = self.plot_confusion_matrix()
        
        # Error analysis
        print("\n=== Error Analysis ===")
        error_df, error_summary = self.error_analysis()
        print("Error Summary by Class:")
        print(error_summary)
        fig3 = self.plot_error_rates()
        
        # Class similarity
        print("\n=== Class Similarity Analysis ===")
        similarity = self.class_similarity_analysis()
        print(similarity)
        fig4 = self.plot_class_similarity()
        
        # Most comfused pairs
        print("\n=== Most Confused Pairs ===")
        most_confused_pairs = self.find_most_confused_pairs()
        print(most_confused_pairs)
        
        # Social group confusion analysis
        print("\n=== Social Group Confusion Analysis ===")
        # social_group_results = self.social_group_confusion_analysis()
        social_group_results = self.per_group_confusion_analysis(group_type='social group')
        for group, data in social_group_results.items():
            print(f"\nGroup: {group}")
            print(data['confusion_matrix'])
            print(f"\nConfusion Rate:")
            print(data['confusion_rate'])
            print(f"\nError Distribution:")
            print(data['error_distribution'])
            print(f"Members: {data['members']}")
            print(f"Overall Confusion Rate: {data['overall_rate']:.2f}")
        fig5 = self.plot_group_confusion(group_type='social group', selected_group=None)
        fig7 = self.plot_group_confusion(group_type='social group', selected_group=None, metric='error_distribution')
        
        # Gender group confusion analysis
        print("\n=== Gender Group Confusion Analysis ===")
        social_group_results = self.per_group_confusion_analysis(group_type='gender group')
        for group, data in social_group_results.items():
            print(f"\nGroup: {group}")
            print(data['confusion_matrix'])
            print(f"\nConfusion Rate:")
            print(data['confusion_rate'])
            print(f"\nError Distribution:")
            print(data['error_distribution'])
            print(f"Members: {data['members']}")
            print(f"Overall Confusion Rate: {data['overall_rate']:.2f}")
        fig6 = self.plot_group_confusion(group_type='gender group', selected_group=None)
    
        # Save figures if output_dir is provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            fig1.savefig(os.path.join(output_dir, 'class_distribution.png'))
            fig2.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            fig3.savefig(os.path.join(output_dir, 'error_rates.png'))
            fig4.savefig(os.path.join(output_dir, 'class_similarity.png'))
            fig5.savefig(os.path.join(output_dir, 'social_group_confusion.png'))
            fig6.savefig(os.path.join(output_dir, 'gender_group_confusion.png'))
            fig7.savefig(os.path.join(output_dir, 'social_group_error_distribution.png'))
        
            
            # Save metrics to CSV
            class_metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
            location_metrics_df.to_csv(os.path.join(output_dir, 'per_location_metrics.csv'), index=False)
            pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
            dist_df.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
            error_summary.to_csv(os.path.join(output_dir, 'error_summary.csv'))
            similarity.to_csv(os.path.join(output_dir, 'class_similarity.csv'))
            most_confused_pairs.to_csv(os.path.join(output_dir, 'most_confused_pairs.csv'), index=False)                
            
            print(f"Report saved to {output_dir}")
        
        plt.show()
        
        return {
            'class_metrics_df': class_metrics_df,
            'location_metrics_df': location_metrics_df,
            'summary': summary,
            'dist_df': dist_df,
            'error_df': error_df,
            'error_summary': error_summary,
            'similarity': similarity,
            'most_confused_pairs': most_confused_pairs,
            'social_group_results': social_group_results,
        }
        
        
# visualization with vx51
def visualize_embeddings(embeddings, actuals, predictions, confidence_scores, label_names, img_paths, locations):
    """
    Visualize embeddings with FiftyOne, coloring points by labels and predictions
    
    Args:
        embeddings: numpy array of embeddings
        actuals: ground truth actuals (indices)
        predictions: predicted labels (indices)
        confidence_scores: confidence of predictions
        val_dataset: the validation dataset
        img_paths: paths to the images (if available)
        label_names: names of the labels
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)    
    # Create a new FiftyOne dataset    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = fo.Dataset(f"embedding_visualization_{timestamp}")
    dataset.persistent = True  # Make the dataset persistent
    
    # Convert labels and predictions to readable form if label_names provided
    readable_labels = [label_names[l] for l in actuals]
    readable_predictions = [label_names[p] for p in predictions]

    # Calculate if prediction is correct
    is_correct = [pred == actual for pred, actual in zip(predictions, actuals)]
    
    # Add samples to dataset
    for i, (emb, reduced_embedding, label, pred, score, correct, img_path, location) in enumerate(
        zip(embeddings, reduced_embeddings, readable_labels, readable_predictions, confidence_scores, is_correct, img_paths, locations)
    ):
        sample = fo.Sample(filepath=img_path)

        # Add metadata
        sample["embedding"] = emb.tolist()  # Convert to list for FiftyOne
        sample["reduced_embedding"] = reduced_embedding.tolist()
        sample["ground_truth"] = label
        sample["prediction"] = pred
        sample["confidence"] = float(score)
        sample["correct"] = correct
        sample["location"] = location
        # sample["sample_id"] = i
        
        # Add to dataset
        dataset.add_sample(sample)
    
    # Compute embedding visualization
    results = fob.compute_visualization(
        dataset,
        embeddings="embedding",
        method="umap",  # You can also try "tsne" or "pca"
        brain_key="embedding_viz"
    )
    
    # Launch FiftyOne App to view the visualization
    session = fo.launch_app(dataset)

    # Add the views to the session
    session.dataset = dataset
    session.wait()  # Wait for the session to finish
    
    print(f"Total samples: {len(dataset)}")
    
    return session

# qualitative evaluation function
# Evaluate a model using Cross-Video KNN classification
def qual_eval_model(MODEL_PATH, query_loader, query_dataset, DATA_PATH_51, config, gallery_loader, verbose=False, visualize=False):
    torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking

    # Initialize Timm Wrapper Model & properly initialize the linear layers
    embedding_size = 256
    dropout_p = 0.0
    embedding_id = "linear"
    pool_mode = "none"
    embedding_model = TimmWrapper(
        backbone_name='vit_large_patch14_dinov2.lvd142m',
        embedding_size=embedding_size,
        embedding_id=embedding_id,
        dropout_p=dropout_p,
        pool_mode=pool_mode,
        img_size=224
    ).to(DEVICE)

    # Load the model weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    # remove the prefix "module." from the keys of the state_dict
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(memory_format=torch.channels_last)
    embedding_model.eval()

    with torch.amp.autocast(device_type='cuda'):
        query_embeddings, query_labels, query_video_ids = generate_embeddings(embedding_model, query_loader, config, use_amp=True)
        embeddings, actuals, predictions, confidence_scores = KNN_qual_eval(query_embeddings, query_dataset.images_for_cv_knn, query_labels, query_video_ids, config, gallery_loader, embedding_model, distance_metric="euclidean")
    
    # get the label names from the val_dataset
    label_names = list(val_dataset.idx_to_label.values()) # 
    # temp: remove the "val/" in label names ()
    label_names = [name.split("/")[1] for name in label_names]
    
    # get the camera locations from the val_dataset
    locations = val_dataset.locations
    print(locations[0])
    
    evaluator = GorillaReIDEvaluator(actuals, predictions, confidence_scores, label_names, locations)
    
    if verbose:
        report = evaluator.generate_full_report(output_dir=EVALUATR_RESULTS_PATH)
        
    if visualize:
        root_dir = DATA_PATH_51
        img_paths = get_image_paths(root_dir)
        img_paths = [os.path.join(root_dir, img_path) for img_path in img_paths]    
        visualize_embeddings(embeddings, actuals, predictions, confidence_scores, label_names, img_paths, locations)


    return evaluator