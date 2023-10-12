% % Metrics for Feature1 vs. Joint 1 Position
% mae_feature1 = 0.2949;
% rmse_feature1 = 0.3762;
% corr_feature1 = 0.7674;
% 
% % Metrics for Feature2 vs. Joint 2 Position
% mae_feature2 = 0.3222;
% rmse_feature2 = 0.4072;
% corr_feature2 = 0.7093;
% 
% % Define group labels
% comparisons = {'Feature1 vs. Joint 1', 'Feature2 vs. Joint 2'};
% 
% % Define metric labels
% metric_labels = {'MAE', 'RMSE', 'Correlation (r)'};
% % Define the metrics for Feature1 vs. Joint 1 Position
% metrics_feature1 = [mae_feature1, rmse_feature1, corr_feature1];
% 
% % Define the metrics for Feature2 vs. Joint 2 Position
% metrics_feature2 = [mae_feature2, rmse_feature2, corr_feature2];
% 
% % Define the significance level (alpha)
% alpha = 0.05;  % You can adjust this as needed
% 
% % Create a figure
% figure;
% 
% % Create subplots for each metric
% for i = 1:3  % Iterate through the three metrics (MAE, RMSE, Correlation)
%     subplot(1, 3, i);  % Create one row with three columns of subplots
%     bar([metrics_feature1(i), metrics_feature2(i)]);
%     % Perform a statistical test (Wilcoxon signed-rank test) for each pair of electrodes
%     [p_value_m1, ~] = ranksum(metrics_feature1(i), metrics_feature2(i));
%     % % Perform a t-test to check for significance
%     % [h, p] = ttest2(metrics_feature1(:,i), metrics_feature2(:,i), 'Alpha', alpha);
%     % 
%     % % Add significance indicator above the bars
%     % if h  % If there is a significant difference
%     %     text(1, metrics_feature1(i), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     %     text(2, metrics_feature2(i), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     % end
% 
%     % Add an indicator if there's a significant difference
%     if p_value_m1 < 0.05 % Set your significance level here
%         text(1, metrics_feature1(i), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%         text(2, metrics_feature2(i), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     end
% 
%     title(sprintf('%s Comparison', metric_labels{i}));
%     ylabel('Metric Value');
%     set(gca, 'XTickLabel', {'Feature1 vs. Joint 1', 'Feature2 vs. Joint 2'});
% end
% 
% % Set a common title for all subplots
% sgtitle('Metrics Comparison: Feature1 vs. Joint 1 and Feature2 vs. Joint 2');
% -----------------------------------
% % Define the metrics for Feature1 vs. Joint 1 Position
% mae_feature1 = 0.2949;
% rmse_feature1 = 0.3762;
% corr_feature1 = 0.7674;
% 
% % Define the metrics for Feature2 vs. Joint 2 Position
% mae_feature2 = 0.3222;
% rmse_feature2 = 0.4072;
% corr_feature2 = 0.7093;
% 
% % Define metric labels
% metric_labels = {'MAE', 'RMSE', 'Correlation (r)'};
% 
% alpha = 0.05;
% % Create a figure
% figure;
% 
% % Create subplots for each metric
% for i = 1:3  % Iterate through the three metrics (MAE, RMSE, Correlation)
%     subplot(1, 3, i);  % Create one row with three columns of subplots
% 
%     % Organize the data into vectors for each metric
%     data_feature1 = [mae_feature1, rmse_feature1, corr_feature1];
%     data_feature2 = [mae_feature2, rmse_feature2, corr_feature2];
% 
%     % Perform a t-test to check for significance
%     [h, p] = ttest2(data_feature1(i), data_feature2(i), 'Alpha', alpha);
% 
%     % Create bar plot
%     bar([data_feature1(i), data_feature2(i)]);
% 
%     % Add significance indicator above the bars
%     if h  % If there is a significant difference
%         x_centers = [1, 2];
%         y_max = max([data_feature1(i), data_feature2(i)]) + 0.1 * max([data_feature1(i), data_feature2(i)]);
%         text(x_centers, repmat(y_max, 1, 2), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     end
% 
%     title(sprintf('%s Comparison', metric_labels{i}));
%     ylabel('Metric Value');
%     set(gca, 'XTickLabel', {'Feature1 vs. Joint 1', 'Feature2 vs. Joint 2'});
% end
% 
% % Set a common title for all subplots
% sgtitle('Metrics Comparison: Feature1 vs. Joint 1 and Feature2 vs. Joint 2');
% --------------------------------------------------
% Define the metrics for Feature1 vs. Joint 1 Position
metrics_feature1 = [0.2949, 0.3762, 0.7674];

% Define the metrics for Feature2 vs. Joint 2 Position
metrics_feature2 = [0.3222, 0.4072, 0.7093];

% Define metric labels
metric_labels = {'MAE', 'RMSE', 'Correlation (r)'};

% Define the significance level (alpha)
alpha = 0.05;  % You can adjust this as needed

% Create a figure
figure;

% Create subplots for each metric
for i = 1:3  % Iterate through the three metrics (MAE, RMSE, Correlation)
    subplot(1, 3, i);  % Create one row with three columns of subplots
    
    % Perform a t-test to check for significance
    [h, p] = ranksum(metrics_feature1(i), metrics_feature2(i));
    
    % Create bar plot
    bar([metrics_feature1(i), metrics_feature2(i)]);
    
    % Add significance indicator above the bars
    if h  % If there is a significant difference
        x_centers = [1, 2];
        y_max = max([metrics_feature1(i), metrics_feature2(i)]) + 0.1 * max([metrics_feature1(i), metrics_feature2(i)]);
        text(x_centers, repmat(y_max, 1, 2), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
    end
    
    title(sprintf('%s Comparison', metric_labels{i}));
    ylabel('Metric Value');
    % set(gca, 'XTickLabel', {'Feature1 vs. Joint 1', 'Feature2 vs. Joint 2'});
end

% Set a common title for all subplots
sgtitle('Metrics Comparison: Feature1 vs. Joint 1 and Feature2 vs. Joint 2');


% ----------------------------
% % Define the metrics for Feature1 vs. Joint 1 Position
% metrics_feature1 = [mae_feature1, rmse_feature1, corr_feature1];
% 
% % Define the metrics for Feature2 vs. Joint 2 Position
% metrics_feature2 = [mae_feature2, rmse_feature2, corr_feature2];
% 
% % Define the significance level (alpha)
% alpha = 0.05;  % You can adjust this as needed
% 
% % Create a figure
% figure;
% 
% % Create subplots for each metric
% for i = 1:3  % Iterate through the three metrics (MAE, RMSE, Correlation)
%     subplot(1, 3, i);  % Create one row with three columns of subplots
% 
%     % Perform a t-test to check for significance
%     [h, p] = ttest2(metrics_feature1(:, i), metrics_feature2(:, i), 'Alpha', alpha);
% 
%     % Organize the data into a cell array for boxplot
%     data = [metrics_feature1(:, i), metrics_feature2(:, i)];
% 
%     % Create the boxplot
%     % boxplot(data, 'Labels');
%      boxplot(data, ...
%     'Labels', {'Join 2 Position', 'Neural Feature 2'});
% 
%     % Add significance indicator above the boxes
%     if h  % If there is a significant difference
%         x_centers = [1, 2];
%         y_max = max(max(cell2mat(data)) + 0.1 * max(max(cell2mat(data))));
%         text(x_centers, repmat(y_max, 1, 2), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     end
% 
%     title(sprintf('%s Comparison', metric_labels{i}));
%     ylabel('Metric Value');
% end
% 
% % Set a common title for all subplots
% sgtitle('Metrics Comparison: Feature1 vs. Joint 1 and Feature2 vs. Joint 2');
% ----------------------------------------------
% % Define the metrics for Feature1 vs. Joint 1 Position
% metrics_feature1 = [mae_feature1, rmse_feature1, corr_feature1];
% 
% % Define the metrics for Feature2 vs. Joint 2 Position
% metrics_feature2 = [mae_feature2, rmse_feature2, corr_feature2];
% 
% % Define the significance level (alpha)
% alpha = 0.05;  % You can adjust this as needed
% 
% % Create a figure
% figure;
% 
% % Create subplots for each metric
% for i = 1:3  % Iterate through the three metrics (MAE, RMSE, Correlation)
%     subplot(1, 3, i);  % Create one row with three columns of subplots
% 
%     % Perform a t-test to check for significance
%     [h, p] = ttest2(metrics_feature1(:, i), metrics_feature2(:, i), 'Alpha', alpha);
% 
%     % Organize the data into a cell array for boxplot
%     data = [metrics_feature1(:, i), metrics_feature2(:, i)];
% 
%     % Create the boxplot
%     boxplot(data, ...
%     'Labels', {'Join 2 Position', 'Neural Feature 2'});
% 
%     % Add significance indicator above the boxes
%     if h  % If there is a significant difference
%         x_centers = [1, 2];
%         y_max = max(max(cell2mat(data)) + 0.1 * max(max(cell2mat(data))));
%         text(x_centers, repmat(y_max, 1, 2), '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
%     end
% 
%     title(sprintf('%s Comparison', metric_labels{i}));
%     ylabel('Metric Value');
% end
% 
% % Set a common title for all subplots
% sgtitle('Metrics Comparison: Feature1 vs. Joint 1 and Feature2 vs. Joint 2');


