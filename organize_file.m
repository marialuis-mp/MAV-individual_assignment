function [file_organized] = organize_file(File, number_rows)
%Organizes file:
    % All the information regarding a image is put in the same row
    % Each row has a M x 8 matrix in which M is the number of gates and
    % the 8 points are the coordinates of each gate
    file_organized = table;
    box = zeros(4,8);
    j=1;
    Number_of_pictures=1;
    for i=1:number_rows-1
        % extract x1, y1, w and h from row
        box(j,:)=File{i,2:end};
        if strcmp(File.Var1(i),File.Var1(i+1))
            j=j+1;
        else
            gate = zeros(j,8);
            for k = 1:j
                gate(k,:)=box(k,:);
            end
            j=1;
            file_organized(Number_of_pictures, :)={File.Var1(i), {gate}};
            Number_of_pictures=Number_of_pictures+1;
        end
    end
    i=i+1;
    box(j,:)=File{i,2:end};
    gate = zeros(j,8);
    for k = 1:j
        gate(k,:)=box(k,:);
    end
    file_organized(Number_of_pictures, :)={File.Var1(i), {gate}};
end

