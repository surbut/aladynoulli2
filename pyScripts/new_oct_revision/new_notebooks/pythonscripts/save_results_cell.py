# ===== SAVE RESULTS TO DISK (to avoid rerunning long computation) =====
# Paste this cell right after line 157 (after the print statements)

print(f"\n{'='*80}")
print("SAVING RESULTS TO DISK")
print(f"{'='*80}")

results_dir = '/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/saved_results/'
import os
os.makedirs(results_dir, exist_ok=True)

# Save all 6 result lists
print("\nSaving Fixed Enrollment results...")
torch.save(fixed_enrollment_10yr_results, f'{results_dir}fixed_enrollment_10yr_results.pt')
torch.save(fixed_enrollment_30yr_results, f'{results_dir}fixed_enrollment_30yr_results.pt')
torch.save(fixed_enrollment_static_10yr_results, f'{results_dir}fixed_enrollment_static_10yr_results.pt')

print("Saving Fixed Retrospective results...")
torch.save(fixed_retrospective_10yr_results, f'{results_dir}fixed_retrospective_10yr_results.pt')
torch.save(fixed_retrospective_30yr_results, f'{results_dir}fixed_retrospective_30yr_results.pt')
torch.save(fixed_retrospective_static_10yr_results, f'{results_dir}fixed_retrospective_static_10yr_results.pt')

print(f"\n✓ All results saved to {results_dir}")
print(f"  - fixed_enrollment_10yr_results.pt ({len(fixed_enrollment_10yr_results)} batches)")
print(f"  - fixed_enrollment_30yr_results.pt ({len(fixed_enrollment_30yr_results)} batches)")
print(f"  - fixed_enrollment_static_10yr_results.pt ({len(fixed_enrollment_static_10yr_results)} batches)")
print(f"  - fixed_retrospective_10yr_results.pt ({len(fixed_retrospective_10yr_results)} batches)")
print(f"  - fixed_retrospective_30yr_results.pt ({len(fixed_retrospective_30yr_results)} batches)")
print(f"  - fixed_retrospective_static_10yr_results.pt ({len(fixed_retrospective_static_10yr_results)} batches)")

print(f"\n{'='*80}")
print("To reload later, use:")
print(f"{'='*80}")
print("fixed_enrollment_10yr_results = torch.load('{results_dir}fixed_enrollment_10yr_results.pt')")
print("fixed_enrollment_30yr_results = torch.load('{results_dir}fixed_enrollment_30yr_results.pt')")
print("fixed_enrollment_static_10yr_results = torch.load('{results_dir}fixed_enrollment_static_10yr_results.pt')")
print("fixed_retrospective_10yr_results = torch.load('{results_dir}fixed_retrospective_10yr_results.pt')")
print("fixed_retrospective_30yr_results = torch.load('{results_dir}fixed_retrospective_30yr_results.pt')")
print("fixed_retrospective_static_10yr_results = torch.load('{results_dir}fixed_retrospective_static_10yr_results.pt')")

