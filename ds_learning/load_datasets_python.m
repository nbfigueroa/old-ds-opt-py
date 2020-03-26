function [demos, data] = load_datasets_python(dataset_name, sub_sample)

data_python  = load(dataset_name);
demos    = [];
N_demos = length(unique(data_python.labels));
for l=1:N_demos
    % Gather Data
    ids = find(data_python.labels == l-1);
    demos{l}.x = data_python.x(1,ids);
    demos{l}.y = data_python.y(1,ids);
    demos{l}.t = abs(data_python.timestamps(1,ids));
end

data = []; dt = [];
for dem = 1:N_demos
    x_obs_dem = [demos{dem}.x; demos{dem}.y]';
    dt = [dt mean(diff(demos{dem}.t'))]; % Average sample time
    dx_nth = sgolay_time_derivatives(x_obs_dem, dt(dem), 2, 3, 7);
    data{dem} = [dx_nth(1:sub_sample:end,:,1),dx_nth(1:sub_sample:end,:,2)]';    
end

end