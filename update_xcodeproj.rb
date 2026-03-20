require 'xcodeproj'
require 'fileutils'

project_path = '/Users/omkarthawakar/MyApp/VisualSeek/VisualSeek/VisualSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)
target = project.targets.find { |t| t.name == 'VisualSeek' }

if target.nil?
  puts "Target 'VisualSeek' not found!"
  exit(1)
end

main_group = project.main_group.find_subpath('VisualSeek', true) # Get or create VisualSeek group

# Clear all existing files from the main VisualSeek group and target
main_group.clear()
target.source_build_phase.files_references.each do |ref|
  if ref.path && ref.path.start_with?('VisualSeek/')
      target.source_build_phase.remove_file_reference(ref)
  end
end

source_dir = '/Users/omkarthawakar/MyApp/VisualSeek/VisualSeek/VisualSeek'

def add_files_to_group(project, target, group, dir_path)
  Dir.foreach(dir_path) do |entry|
    next if entry == '.' or entry == '..' or entry == '.DS_Store'
    
    full_path = File.join(dir_path, entry)
    
    if File.directory?(full_path)
      if entry.end_with?('.xcassets')
        # Add assets
        file_ref = group.new_reference(entry)
        target.resources_build_phase.add_file_reference(file_ref, true)
      else
        # It's a directory, create a subgroup and recurse
        subgroup = group.children.find { |c| c.display_name == entry || c.path == entry }
        subgroup ||= group.new_group(entry, entry)
        add_files_to_group(project, target, subgroup, full_path)
      end
    else
      # It's a file
      if entry == 'Info.plist'
        # Do not add Info.plist to build phases normally, but add to project group
        file_ref = group.new_reference(entry)
      elsif entry.end_with?('.swift') || entry.end_with?('.m') || entry.end_with?('.cpp') || entry.end_with?('.c')
        file_ref = group.new_reference(entry)
        target.source_build_phase.add_file_reference(file_ref, true)
      elsif entry.end_with?('.mlmodel') || entry.end_with?('.mlpackage')
        file_ref = group.new_reference(entry)
        target.source_build_phase.add_file_reference(file_ref, true)
      else
        # Other resources
        file_ref = group.new_reference(entry)
        target.resources_build_phase.add_file_reference(file_ref, true)
      end
    end
  end
end

puts "Adding files to project..."
add_files_to_group(project, target, main_group, source_dir)

# Special sync for ContentView.swift, VisualSeekApp.swift, etc. if old ones were present, let's just make sure we didn't miss them if they are in App/
# The script recursively adds everything. 

# Let's also remove any stale references from the target to avoid build errors.
target.source_build_phase.files.each do |build_file|
  if !build_file.file_ref || !File.exist?(build_file.file_ref.real_path.to_s)
     target.source_build_phase.remove_build_file(build_file)
  end
end
target.resources_build_phase.files.each do |build_file|
  if !build_file.file_ref || !File.exist?(build_file.file_ref.real_path.to_s)
     target.resources_build_phase.remove_build_file(build_file)
  end
end

project.save
puts "Successfully updated project!"
