#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <time.h>

namespace fs = std::filesystem;

#define INPUT_NODES  784  // 28x28 pixels
#define OUTPUT_NODES 10
#define TRAIN_IMAGES  8700  // 28x28 pixels
#define TEST_IMAGES  1300   // 28x28 pixels

const char* train_path = "C:\\Users\\HP\\Desktop\\vs file\\CNN\\data\\training";
const char* test_path = "C:\\Users\\HP\\Desktop\\vs file\\CNN\\data\\testing";

double sigmoid(double x) {             //  ���򴫲�ʱ sigmoid�����
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {  //  ���򴫲�ʱʹ�� sigmoid�ĵ���
    return x * (1.0 - x);
}

void initialize_weights(double* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

void shuffle_data(double images[][INPUT_NODES], int labels[], int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int j = rand() % num_samples;  // �������
        // ����ͼ��
        for (int k = 0; k < INPUT_NODES; k++) {
            double temp = images[i][k];
            images[i][k] = images[j][k];
            images[j][k] = temp;
        }
        // ������ǩ
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}

//  ǰ�򴫲�
void forward_propagation(double* input, double* hidden1, double* hidden2, double* output,
    double* input_hidden1_weights, double* hidden1_hidden2_weights, double* hidden2_output_weights,
    int HIDDEN_NODES_1, int HIDDEN_NODES_2, bool apply_dropout = false, double dropout_rate = 0.5) {
    // Dropout mask
    int* dropout_mask1 = (int*)malloc(HIDDEN_NODES_1 * sizeof(int));
    int* dropout_mask2 = (int*)malloc(HIDDEN_NODES_2 * sizeof(int));
    for (int i = 0; i < HIDDEN_NODES_1; i++) {
        dropout_mask1[i] = (rand() / (double)RAND_MAX) > dropout_rate ? 1 : 0;
    }
    for (int i = 0; i < HIDDEN_NODES_2; i++) {
        dropout_mask2[i] = (rand() / (double)RAND_MAX) > dropout_rate ? 1 : 0;
    }

    // ����㵽���ز�1
    for (int i = 0; i < HIDDEN_NODES_1; i++) {
        hidden1[i] = 0.0;
        for (int j = 0; j < INPUT_NODES; j++) {// ���½ڵ�ֵ
            hidden1[i] += input[j] * input_hidden1_weights[j * HIDDEN_NODES_1 + i];
        }
        hidden1[i] = sigmoid(hidden1[i]);
        // Ӧ��dropout
        if (apply_dropout) hidden1[i] *= dropout_mask1[i];
    }

    // ���ز�1�����ز�2
    for (int i = 0; i < HIDDEN_NODES_2; i++) {
        hidden2[i] = 0.0;
        for (int j = 0; j < HIDDEN_NODES_1; j++) {
            hidden2[i] += hidden1[j] * hidden1_hidden2_weights[j * HIDDEN_NODES_2 + i];
        }
        hidden2[i] = sigmoid(hidden2[i]);
        // Ӧ��dropout
        if (apply_dropout) hidden2[i] *= dropout_mask2[i];
    }

    // ���ز�2�������
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output[i] = 0.0;
        for (int j = 0; j < HIDDEN_NODES_2; j++) {
            output[i] += hidden2[j] * hidden2_output_weights[j * OUTPUT_NODES + i];
        }
        output[i] = sigmoid(output[i]);
    }

    free(dropout_mask1);
    free(dropout_mask2);
}

void back_propagation(double* input, double* hidden1, double* hidden2, double* output, double* target,
    double* input_hidden1_weights, double* hidden1_hidden2_weights, double* hidden2_output_weights,
    int HIDDEN_NODES_1, int HIDDEN_NODES_2, double LEARNING_RATE, double L2_REGULARIZATION) {
    double output_error[OUTPUT_NODES];
    double* hidden2_error = (double*)malloc(HIDDEN_NODES_2 * sizeof(double));
    double* hidden1_error = (double*)malloc(HIDDEN_NODES_1 * sizeof(double));

    // ��������
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output_error[i] = target[i] - output[i];  //��������ʧ����
    }

    // ���ز�2���
    for (int i = 0; i < HIDDEN_NODES_2; i++) {
        hidden2_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden2_error[i] += output_error[j] * hidden2_output_weights[i * OUTPUT_NODES + j];
        }
        hidden2_error[i] *= sigmoid_derivative(hidden2[i]);
    }

    // ���ز�1���
    for (int i = 0; i < HIDDEN_NODES_1; i++) {
        hidden1_error[i] = 0.0;
        for (int j = 0; j < HIDDEN_NODES_2; j++) {
            hidden1_error[i] += hidden2_error[j] * hidden1_hidden2_weights[i * HIDDEN_NODES_2 + j];
        }
        hidden1_error[i] *= sigmoid_derivative(hidden1[i]);
    }

    // ����Ȩ�أ����ز�2������㣩
    for (int i = 0; i < HIDDEN_NODES_2; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {           //�������ڵ����ӵ����ز�2���߶�Ҫ����Ȩ��
                                                           //Ȩ�غ����������
            hidden2_output_weights[i * OUTPUT_NODES + j] += LEARNING_RATE * output_error[j] * hidden2[i];
            // L2 ���򻯣�����Ȩ��ƽ���͵ĳͷ��
            hidden2_output_weights[i * OUTPUT_NODES + j] -= LEARNING_RATE * L2_REGULARIZATION * hidden2_output_weights[i * OUTPUT_NODES + j];
        }                                                 
    }

    // ����Ȩ�أ����ز�1�����ز�2��
    for (int i = 0; i < HIDDEN_NODES_1; i++) {
        for (int j = 0; j < HIDDEN_NODES_2; j++) {
            hidden1_hidden2_weights[i * HIDDEN_NODES_2 + j] += LEARNING_RATE * hidden2_error[j] * hidden1[i];
            // L2 ����
            hidden1_hidden2_weights[i * HIDDEN_NODES_2 + j] -= LEARNING_RATE * L2_REGULARIZATION * hidden1_hidden2_weights[i * HIDDEN_NODES_2 + j];
        }                                //L2_REGULARIZATIONΪ0.1~0.0001�������Լ�С ��Ȩ�ؽڵ��Ӱ��
    }

    // ����Ȩ�أ�����㵽���ز�1��
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES_1; j++) {
            input_hidden1_weights[i * HIDDEN_NODES_1 + j] += LEARNING_RATE * hidden1_error[j] * input[i];
            // L2 ����
            input_hidden1_weights[i * HIDDEN_NODES_1 + j] -= LEARNING_RATE * L2_REGULARIZATION * input_hidden1_weights[i * HIDDEN_NODES_1 + j];
        }
    }

    free(hidden2_error);
    free(hidden1_error);
}

// ��������
void load_data(const std::string& dataset_path, double images[][INPUT_NODES], int labels[], int max_images) {
    int index = 0;

    for (int label = 0; label <= 9 && index < max_images; ++label) {
        std::string label_path = dataset_path + "\\" + std::to_string(label);
        if (!fs::exists(label_path) || !fs::is_directory(label_path)) continue;

        for (const auto& entry : fs::directory_iterator(label_path)) {
            if (entry.is_regular_file() && index < max_images) {
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (!img.empty()) {
                    cv::resize(img, img, cv::Size(28, 28));
                    for (int i = 0; i < 28; ++i) {
                        for (int j = 0; j < 28; ++j) {
                            images[index][i * 28 + j] = static_cast<double>(img.at<uchar>(i, j)) / 255.0;//��һ������
                        }
                    }
                    labels[index++] = label;
                }
            }
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));

    // �ֶ����볬����
    int HIDDEN_NODES_1, HIDDEN_NODES_2, EPOCHS;
    double LEARNING_RATE, L2_REGULARIZATION, DROPOUT_RATE;

    printf("���������ز�1�Ľڵ���: ");
    if (scanf_s("%d", &HIDDEN_NODES_1) != 1) {
        printf("�������������һ��������\n");
        return -1;
    }
    printf("���������ز�2�Ľڵ���: ");
    if (scanf_s("%d", &HIDDEN_NODES_2) != 1) {
        printf("�������������һ��������\n");
        return -1;
    }
    printf("������ѧϰ��: ");
    if (scanf_s("%lf", &LEARNING_RATE) != 1) {
        printf("�������������һ����������\n");
        return -1;
    }
    printf("������ѵ������: ");
    if (scanf_s("%d", &EPOCHS) != 1) {
        printf("�������������һ��������\n");
        return -1;
    }
    printf("������L2���򻯲���: ");
    if (scanf_s("%lf", &L2_REGULARIZATION) != 1) {
        printf("�������������һ����������\n");
        return -1;
    }
    printf("������dropout�ʣ�0-1֮�䣩: ");
    if (scanf_s("%lf", &DROPOUT_RATE) != 1) {
        printf("�������������һ����������\n");
        return -1;
    }

    // �洢ÿ��epoch�������
    double* epoch_errors = (double*)malloc(EPOCHS * sizeof(double));

    // ��̬�����ڴ�
    double* input = (double*)malloc(INPUT_NODES * sizeof(double));
    double* hidden1 = (double*)malloc(HIDDEN_NODES_1 * sizeof(double));
    double* hidden2 = (double*)malloc(HIDDEN_NODES_2 * sizeof(double));
    double* output = (double*)malloc(OUTPUT_NODES * sizeof(double));
    double* target = (double*)malloc(OUTPUT_NODES * sizeof(double));

                                     //���Ȩ�����ߣ�ÿ����֮�䶼������ڵ�˻���������
    double* input_hidden1_weights = (double*)malloc(INPUT_NODES * HIDDEN_NODES_1 * sizeof(double));
    double* hidden1_hidden2_weights = (double*)malloc(HIDDEN_NODES_1 * HIDDEN_NODES_2 * sizeof(double));
    double* hidden2_output_weights = (double*)malloc(HIDDEN_NODES_2 * OUTPUT_NODES * sizeof(double));

    double(*train_images)[INPUT_NODES] = (double(*)[INPUT_NODES])malloc(TRAIN_IMAGES * INPUT_NODES * sizeof(double));
    int* train_labels = (int*)malloc(TRAIN_IMAGES * sizeof(int));

    double(*test_images)[INPUT_NODES] = (double(*)[INPUT_NODES])malloc(TEST_IMAGES * INPUT_NODES * sizeof(double));
    int* test_labels = (int*)malloc(TEST_IMAGES * sizeof(int));

    // ��ʼ��Ȩ��
    initialize_weights(input_hidden1_weights, INPUT_NODES * HIDDEN_NODES_1);
    initialize_weights(hidden1_hidden2_weights, HIDDEN_NODES_1 * HIDDEN_NODES_2);
    initialize_weights(hidden2_output_weights, HIDDEN_NODES_2 * OUTPUT_NODES);

    // ��������
    load_data(train_path, train_images, train_labels, TRAIN_IMAGES);
    load_data(test_path, test_images, test_labels, TEST_IMAGES);

    // ѵ������
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_data(train_images, train_labels, TRAIN_IMAGES);

        double total_error = 0.0;  // ��ʼ��ÿ��epoch�������
        int correct_train = 0;  // ����ͳ��ѵ������ȷ��

        for (int i = 0; i < TRAIN_IMAGES; i++) {
            // ���������Ŀ��
            for (int j = 0; j < INPUT_NODES; j++) {
                input[j] = train_images[i][j];
            }
            for (int k = 0; k < OUTPUT_NODES; k++) {
                target[k] = (train_labels[i] == k) ? 1.0 : 0.0;
            }

            // ǰ�򴫲�
            forward_propagation(input, hidden1, hidden2, output, input_hidden1_weights, hidden1_hidden2_weights, hidden2_output_weights, HIDDEN_NODES_1, HIDDEN_NODES_2, true, DROPOUT_RATE);

            // �����������أ�
            double sample_error = 0.0;
            for (int k = 0; k < OUTPUT_NODES; k++) {
                if (target[k] > 0) {
                    // ��ֹlog(0)���������һ����С������1e-15
                    sample_error -= target[k] * log(output[k] + 1e-15);
                }
            }
            total_error += sample_error;
            // ���Ԥ����
            int predicted = 0;
            for (int k = 1; k < OUTPUT_NODES; k++) {
                if (output[k] > output[predicted]) {
                    predicted = k;
                }
            }
            if (predicted == train_labels[i]) {
                correct_train++;
            }

            // ���򴫲�
            back_propagation(input, hidden1, hidden2, output, target, input_hidden1_weights, hidden1_hidden2_weights, hidden2_output_weights, HIDDEN_NODES_1, HIDDEN_NODES_2, LEARNING_RATE, L2_REGULARIZATION);
        }
        // �洢��ǰepoch�����
        epoch_errors[epoch] = total_error;

        // ��ӡ��ǰ epoch ��ѵ������ȷ�ʺ������
        printf("Epoch %d/%d, Training Accuracy: %.2f%%, Total Error: %.5f\n",
            epoch + 1, EPOCHS, (double)correct_train / TRAIN_IMAGES * 100.0, total_error);
    }

    // Ԥ�����洢��excel����
    FILE* prediction_file;
    errno_t pred_err = fopen_s(&prediction_file, "C:\\Users\\HP\\Desktop\\pycharm\\put py here\\machine learning\\prediction_results.csv", "w");
    if (pred_err == 0) {
        fprintf(prediction_file, "����,ʵ�ʱ�ǩ,Ԥ���ǩ\n");
    }
    else {
        printf("�޷���Ԥ�����ļ���\n");
        return -1; // ����ļ��޷��򿪣����˳�
    }

    // ��������
    int correct_test = 0;
    double total_error = 0.0;  // ��ʼ�������
    for (int i = 0; i < TEST_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            input[j] = test_images[i][j];
        }
        forward_propagation(input, hidden1, hidden2, output, input_hidden1_weights, hidden1_hidden2_weights, hidden2_output_weights, HIDDEN_NODES_1, HIDDEN_NODES_2);

        // �����������أ�
        double sample_error = 0.0;
        for (int k = 0; k < OUTPUT_NODES; k++) {
            if (target[k] > 0) {
                // ��ֹlog(0)���������һ����С������1e-15
                sample_error -= target[k] * log(output[k] + 1e-15);
            }
        }
        total_error += sample_error;

        int predicted = 0;
        for (int k = 1; k < OUTPUT_NODES; k++) {
            if (output[k] > output[predicted]) {
                predicted = k;
            }
        }
        // ��¼ʵ�ʱ�ǩ��Ԥ���ǩ�� CSV �ļ�
        fprintf(prediction_file, "%d,%d,%d\n", i + 1, test_labels[i], predicted);

        if (predicted == test_labels[i]) {
            correct_test++;
        }
    }

    printf("Test Accuracy: %.2f%%, Total Error: %.5f\n", (double)correct_test / TEST_IMAGES * 100.0, total_error);

    // �ͷ��ڴ�
    free(input);
    free(hidden1);
    free(hidden2);
    free(output);
    free(target);
    free(input_hidden1_weights);
    free(hidden1_hidden2_weights);
    free(hidden2_output_weights);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    free(epoch_errors);

    fclose(prediction_file);

    return 0;
}
