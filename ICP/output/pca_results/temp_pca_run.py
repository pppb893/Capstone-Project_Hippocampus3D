
import slicer
import os
import time

csv_path = r'D:/capstone/HFH Hippocampus Segmentation_Organized/ICP/output/pca_results/pca_input.csv'
save_path = r'D:/capstone/HFH Hippocampus Segmentation_Organized/ICP/output/pca_results/pca_model.json'

print('--- SlicerSALT PCA Started ---')
print('Input CSV: ' + csv_path)
print('Output JSON: ' + save_path)

parameters = {
    'inputCsv': csv_path,
    'outputJson': save_path,
    'evaluation': 0,
    'shapeNum': 100
}

print('Loading ShapePCA Module and starting computation...')
cli_node = slicer.cli.run(slicer.modules.shapepca, None, parameters)

# Improved Progress Monitoring
last_progress = -1
start_time = time.time()

while cli_node.IsBusy():
    progress = int(cli_node.GetProgress() * 100)
    if progress != last_progress:
        elapsed = int(time.time() - start_time)
        print('  Progress: ' + str(progress) + '% (Elapsed: ' + str(elapsed) + 's)')
        last_progress = progress
    time.sleep(2)

if cli_node.GetStatusString() == 'Completed':
    print('\n--- SUCCESS: PCA Analysis Completed Successfully! ---')
    print('Results saved to: ' + save_path)
else:
    print('\n--- FAILED: PCA Analysis stopped with status: ' + cli_node.GetStatusString() + ' ---')
    print('Error Log: ' + cli_node.GetErrorText())

slicer.util.exit()
