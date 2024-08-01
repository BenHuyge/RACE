%% Reconstruct (continuous) sinogram using SIRT and the cuda projection / backprojection.
%
% If the number of projections in proj_geomS is higher than the number of rows in the sinogram,
% the sinogram is assumed to be continuous and ARTIC is used. Otherwise, plain SIRT is executed.
%
% usage:
% [ vol, pweights, vweights ] = reconstructSIRT( sinogram, proj_geomS, vol_geom, iterations)
% [ vol, pweights, vweights ] = reconstructSIRT( sinogram, proj_geomS, vol_geom, iterations, initialRecon)
% [ vol, pweights, vweights ] = reconstructSIRT( sinogram, proj_geomS, vol_geom, iterations, initialRecon, pweights, vweights)
%
% sinogram:         (continuous) sinogram
% proj_geomS:       projection geometry
% vol_geom:         volume geometry of reconstruction
% iterations:       nb of iterations
% initialRecon:     (optional) initial reconstruction
% pweights:         (optional) projection weights
% vweights:         (optional) volume weights

function [ vol, pweights, vweights, sino_forward ] = reconstructSIRT( sinogram, proj_geomS, vol_geom, iterations, initialRecon, pweights, vweights)

is3d = (length(size(sinogram)) == 3);

% find out sampling ratio (1 = static reconstruction)
vecgeom = strcmp(proj_geomS.type(end-2:end), 'vec');
if vecgeom
    samplingratio = size(proj_geomS.Vectors,1) / size(sinogram,1 + is3d);
else
    samplingratio = length(proj_geomS.ProjectionAngles) / size(sinogram,1 + is3d);
end

if round(samplingratio) ~= samplingratio
    error('#rows in sinogram should be multiple of #projections!');    
end


if exist('initialRecon', 'var')
    vol = initialRecon;
elseif is3d
    vol = zeros(vol_geom.GridColCount, vol_geom.GridRowCount, vol_geom.GridSliceCount);
else
    vol = zeros(vol_geom.GridRowCount, vol_geom.GridColCount);
end

%% implementation with continuous forward and backward projector
% projection weights
if ~exist('pweights', 'var')
    if is3d
        pweights = astra_create_sino3d_cudaHR(ones(size(vol)), proj_geomS, vol_geom);
        vweights = astra_create_backprojection3d_cudaHR(ones(size(pweights)), proj_geomS, vol_geom);
    else
        pweights = astra_create_sino_cudaHR(ones(size(vol)), proj_geomS, vol_geom);
        vweights = astra_create_backprojection_cudaHR(ones(size(pweights)), proj_geomS, vol_geom);
    end
end

if samplingratio > 1
    sino_forward = zeros(size(sinogram));
end

% reconstruction loop
if is3d
    fprintf('Reconstructing ');
    for it = 1: iterations
        fprintf('.');
        % 1. create (continuous) forward projection of vol and difference with sinogram       
        sino_sampled = astra_create_sino3d_cudaHR(vol, proj_geomS, vol_geom);    
        
        if samplingratio > 1
            for c = 1: size(sinogram,2)
                psum = sum(sino_sampled(:, (c-1)*samplingratio +1 : c*samplingratio, :),2)./samplingratio;    
                sino_forward(:,c,:) = psum;
            end

            sino_diff = zeros(size(sinogram) .* [1 samplingratio 1]);

            for c = 1: size(sino_forward,2)
                start = (c-1)*samplingratio;
                sino_diff(:,start + 1 : start + samplingratio, :) = repmat(sinogram(:,c,:)-sino_forward(:,c,:), [1, samplingratio, 1]);
            end
        else
            sino_diff = sinogram - sino_sampled;
            sino_forward = sino_sampled;
        end

        % 3. backproject upsampled diff            
        diff = sino_diff./pweights;
        diff(isnan(diff)) = 0;
        diff(isinf(diff)) = 0;

        diff_vol = astra_create_backprojection3d_cudaHR(diff, proj_geomS, vol_geom);
        diff_vol = diff_vol ./ vweights;
        diff_vol(isnan(diff_vol)) = 0;
        diff_vol(isinf(diff_vol)) = 0;

        vol = vol + diff_vol;
    end
    fprintf(' done \n');
else
    for it = 1: iterations
        % 1. create (continuous) forward projection of vol and difference with sinogram        
        sino_sampled = astra_create_sino_cudaHR(vol, proj_geomS, vol_geom);    
        
        if samplingratio > 1
            for c = 1: size(sinogram,1)
                psum = sum(sino_sampled((c-1)*samplingratio +1 : c*samplingratio,:),1)/samplingratio;    
                sino_forward(c,:) = psum;
            end

            sino_diff = zeros(size(sinogram,1)*samplingratio, size(sinogram,2));

            for c = 1: size(sino_forward,1)
                start = (c-1)*samplingratio;
                sino_diff(start + 1 : start + samplingratio, :) = repmat(sinogram(c,:)-sino_forward(c,:), samplingratio, 1);
            end
        else
            sino_diff = sinogram - sino_sampled;
            sino_forward = sino_sampled;
        end

        % 3. backproject upsampled diff            
        diff = sino_diff./pweights;
        diff(isnan(diff)) = 0;
        diff(isinf(diff)) = 0;

        diff_vol = astra_create_backprojection_cudaHR(diff, proj_geomS, vol_geom);
        diff_vol = diff_vol ./ vweights;
        diff_vol(isnan(diff_vol)) = 0;
        diff_vol(isinf(diff_vol)) = 0;

        vol = vol + diff_vol;
    end
end

