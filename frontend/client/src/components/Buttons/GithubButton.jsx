import { Button } from '@mantine/core';
import { GithubIcon } from '@mantinex/dev-icons';

export function GithubButton(props) {
  return (
    <Button
      leftSection={<GithubIcon style={{ width: '1rem', height: '1rem' }} color="black" />}
      variant="default"
      {...props}
    />
  );
}