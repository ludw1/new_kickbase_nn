# Improved Kickbase NN
This repo is a remake of [kickbase_nn](https://github.com/ludw1/kickbase_nn) with improvements across the board. About a year ago, kickbase changed their API significantly, which broke the original code. This version addresses that and includes several enhancements:
- **Updated API Integration**: The code now works with the latest Kickbase API, ensuring compatibility.
- **Enhanced Data Handling**: Improved data fetching and processing for better performance.
- **Refined Model Training and more models**: I have learned a lot about time series prediction and neural networks since I wrote the original code. This version includes better model architectures and training techniques. It also includes more models to compare against.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ludw1/new_kickbase_nn.git
   ```
2. **Install Dependencies**:
   Navigate to the cloned directory and install the required packages, depending on whether you want to fetch data or train models:
   ```bash
   cd new_kickbase_nn/get_data or cd ../train_model
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
    Create a `.env` file in the `get_data` directory with your Kickbase credentials:
    ```
    KICKBASE_USERNAME=your_username
    KICKBASE_PASSWORD=your_password
    ```
4. **Fetch Player Data**:
    Run the data fetching script:
    ```bash
    python main.py
    ```
5. **Train Models**:
    Navigate to the `train_model` directory and run the training script:
    ```bash
    python training.py
    ```
    You can adjust configurations in `config.py` as needed.
6. **Evaluate Models**:
    After training, evaluate the models using:
    ```bash
    python evaluation.py
    ```

