function [norm_image] = normalize_scan(process_img)
    
    minA = min(min(min(process_img)));
    maxA = max(max(max(process_img)));
    norm_image = (process_img - minA)/(maxA - minA);
   
end